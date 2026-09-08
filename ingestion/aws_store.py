"""Durable GovCloud ingestion store for source objects, manifests, and search."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from datetime import UTC, datetime
from typing import Any

from ingestion.chunking import normalize_document
from ingestion.models import ChangeEvent, ChangeKind
from ingestion.store import ApplyResult
from knowledge.index_opensearch import (
    DEFAULT_DIMENSIONS,
    SigV4OpenSearchClient,
    index_records,
    titan_embedder,
)


class AwsIngestionStore:
    """Idempotently persist source versions and ACL-bearing retrieval chunks."""

    def __init__(
        self,
        *,
        document_table: Any,
        change_table: Any,
        s3_client: Any,
        source_bucket: str,
        kms_key_arn: str,
        search_client: SigV4OpenSearchClient,
        search_index: str,
        region: str,
        security_domain: str,
        max_chunk_chars: int = 1_200,
        embedder: Any | None = None,
    ) -> None:
        self.document_table = document_table
        self.change_table = change_table
        self.s3_client = s3_client
        self.source_bucket = source_bucket
        self.kms_key_arn = kms_key_arn
        self.search_client = search_client
        self.search_index = search_index
        self.security_domain = security_domain
        self.max_chunk_chars = max_chunk_chars
        self.embedder = embedder or titan_embedder(region)

    @classmethod
    def from_environment(cls, *, max_chunk_chars: int = 1_200) -> AwsIngestionStore:
        import os

        import boto3

        required = {
            "DOCUMENT_REGISTRY_TABLE": os.getenv("DOCUMENT_REGISTRY_TABLE", ""),
            "CHANGE_EVENT_TABLE": os.getenv("CHANGE_EVENT_TABLE", ""),
            "SOURCE_BUCKET": os.getenv("SOURCE_BUCKET", ""),
            "J2_KMS_KEY_ARN": os.getenv("J2_KMS_KEY_ARN", ""),
            "OPENSEARCH_ENDPOINT": os.getenv("OPENSEARCH_ENDPOINT", ""),
            "ENTERPRISE_INDEX": os.getenv(
                "ENTERPRISE_INDEX",
                "j2-enterprise-intelligence-v1",
            ),
            "AWS_REGION": os.getenv("AWS_REGION", "us-gov-west-1"),
            "SECURITY_DOMAIN": os.getenv("SECURITY_DOMAIN", "demo"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Durable ingestion configuration is incomplete: {', '.join(missing)}")
        dynamodb = boto3.resource("dynamodb", region_name=required["AWS_REGION"])
        return cls(
            document_table=dynamodb.Table(required["DOCUMENT_REGISTRY_TABLE"]),
            change_table=dynamodb.Table(required["CHANGE_EVENT_TABLE"]),
            s3_client=boto3.client("s3", region_name=required["AWS_REGION"]),
            source_bucket=required["SOURCE_BUCKET"],
            kms_key_arn=required["J2_KMS_KEY_ARN"],
            search_client=SigV4OpenSearchClient(
                required["OPENSEARCH_ENDPOINT"],
                required["AWS_REGION"],
            ),
            search_index=required["ENTERPRISE_INDEX"],
            region=required["AWS_REGION"],
            security_domain=required["SECURITY_DOMAIN"],
            max_chunk_chars=max_chunk_chars,
        )

    def apply(self, events: tuple[ChangeEvent, ...] | list[ChangeEvent]) -> ApplyResult:
        inserted = updated = deleted = unchanged = 0
        records_to_index: list[dict[str, Any]] = []
        for event in events:
            storage_id = self._storage_id(*event.key)
            existing = self.document_table.get_item(
                Key={"document_id": storage_id},
                ConsistentRead=True,
            ).get("Item")
            if event.kind == ChangeKind.DELETE:
                if not existing or existing.get("deleted"):
                    unchanged += 1
                    continue
                self._delete_search_records(event.key[0], event.key[1])
                if existing.get("object_key"):
                    self.s3_client.delete_object(
                        Bucket=self.source_bucket,
                        Key=existing["object_key"],
                    )
                self.document_table.put_item(
                    Item={
                        **existing,
                        "deleted": True,
                        "version": event.version,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                self._write_change("delete", storage_id, event.version)
                deleted += 1
                continue

            if event.document is None:
                raise ValueError("upsert event must include a document")
            document = event.document.normalized()
            fingerprint = self._fingerprint(document)
            if (
                existing
                and not existing.get("deleted")
                and existing.get("fingerprint") == fingerprint
            ):
                unchanged += 1
                continue
            if existing:
                self._delete_search_records(document.corpus, document.document_id)
            object_key = self._object_key(document.corpus, document.document_id, document.version)
            self.s3_client.put_object(
                Bucket=self.source_bucket,
                Key=object_key,
                Body=document.text.encode("utf-8"),
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=self.kms_key_arn,
                ContentType="text/plain; charset=utf-8",
            )
            records = normalize_document(document, max_chars=self.max_chunk_chars)
            records_to_index.extend(record.to_index_dict() for record in records)
            self.document_table.put_item(
                Item={
                    "document_id": storage_id,
                    "corpus_id": document.corpus,
                    "source_document_id": document.document_id,
                    "version": document.version,
                    "content_sha256": document.checksum,
                    "fingerprint": fingerprint,
                    "security_label": document.security_label,
                    "acl_principals": list(document.acl_principals),
                    "record_count": len(records),
                    "object_key": object_key,
                    "deleted": False,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            self._write_change(
                "update" if existing else "insert",
                storage_id,
                document.version,
            )
            if existing:
                updated += 1
            else:
                inserted += 1
        if records_to_index:
            prepared = []
            for record in records_to_index:
                embedding = self.embedder(record["text"])
                if len(embedding) != DEFAULT_DIMENSIONS:
                    raise ValueError("Embedding dimensions do not match the index")
                prepared.append({**record, "embedding": embedding})
            index_records(
                self.search_client,
                prepared,
                self.search_index,
                DEFAULT_DIMENSIONS,
            )
        return ApplyResult(inserted, updated, deleted, unchanged)

    def document_count(self) -> int:
        response = self.document_table.scan(
            Select="COUNT",
            FilterExpression="attribute_not_exists(deleted) OR deleted = :false",
            ExpressionAttributeValues={":false": False},
        )
        count = int(response.get("Count", 0))
        while response.get("LastEvaluatedKey"):
            response = self.document_table.scan(
                Select="COUNT",
                FilterExpression="attribute_not_exists(deleted) OR deleted = :false",
                ExpressionAttributeValues={":false": False},
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            count += int(response.get("Count", 0))
        return count

    def _delete_search_records(self, corpus: str, document_id: str) -> None:
        encoded_index = urllib.parse.quote(self.search_index, safe="-_")
        body = json.dumps(
            {
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"corpus_id": corpus}},
                            {"term": {"document_id": document_id}},
                        ]
                    }
                }
            },
            separators=(",", ":"),
        )
        self.search_client.request(
            "POST",
            f"{encoded_index}/_delete_by_query?refresh=true",
            body,
            "application/json",
        )

    def _write_change(self, kind: str, storage_id: str, version: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        event_id = hashlib.sha256(f"{storage_id}|{version}|{kind}".encode()).hexdigest()
        self.change_table.put_item(
            Item={
                "change_id": event_id,
                "document_id": storage_id,
                "change_kind": kind,
                "detected_at": timestamp,
            }
        )

    def _object_key(self, corpus: str, document_id: str, version: str) -> str:
        document_hash = hashlib.sha256(document_id.encode()).hexdigest()
        version_hash = hashlib.sha256(version.encode()).hexdigest()[:16]
        return f"{self.security_domain}/{corpus}/{document_hash}/{version_hash}.txt"

    @staticmethod
    def _storage_id(corpus: str, document_id: str) -> str:
        return f"{corpus}#{document_id}"

    @staticmethod
    def _fingerprint(document: Any) -> str:
        payload = {
            "version": document.version,
            "checksum": document.checksum,
            "security_label": document.security_label,
            "acl_principals": document.acl_principals,
            "provenance": document.provenance,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
