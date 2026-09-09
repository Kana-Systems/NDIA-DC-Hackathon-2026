"""Tenant/owner-scoped persistent Lens records. No credentials or browser storage."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import HTTPException

from app.config import Settings
from app.models import PrincipalContext


def now():
    return datetime.now(UTC).isoformat()


class WorkspaceStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.s3 = None
        self.search = None
        if settings.workspace_source_bucket:
            import boto3

            self.s3 = boto3.client("s3", region_name=settings.aws_region)
        if settings.workspace_search_enabled:
            from app.workspace_search import WorkspaceSearch

            self.search = WorkspaceSearch(settings)
        self.lock = RLock()
        self.table = None
        self.path = settings.workspace_db_path
        if settings.workspace_table:
            import boto3

            self.table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(
                settings.workspace_table
            )
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS records "
                    "(scope TEXT, id TEXT, data TEXT, PRIMARY KEY(scope,id))"
                )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def scope(principal: PrincipalContext):
        return json.dumps([principal.security_domain, principal.subject], separators=(",", ":"))

    def list(self, principal, kind=None):
        scope = self.scope(principal)
        if self.table is not None:
            from boto3.dynamodb.conditions import Key

            kwargs = {"KeyConditionExpression": Key("scope").eq(scope), "ConsistentRead": True}
            rows = []
            while True:
                page = self.table.query(**kwargs)
                rows.extend(json.loads(item["data"]) for item in page.get("Items", []))
                if not page.get("LastEvaluatedKey"):
                    break
                kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        else:
            with self.connect() as db:
                rows = [
                    json.loads(row[0])
                    for row in db.execute("SELECT data FROM records WHERE scope=?", (scope,))
                ]
        return sorted(
            (r for r in rows if kind is None or r["kind"] == kind),
            key=lambda r: r["updated_at"],
            reverse=True,
        )

    def connections_for_worker(self):
        """Internal scheduler scan; never expose cross-owner rows through an API."""
        if self.table is not None:
            from boto3.dynamodb.conditions import Attr

            args = {"FilterExpression": Attr("kind").eq("connection") | Attr("kind").not_exists()}
            rows = []
            while True:
                page = self.table.scan(**args)
                rows.extend(json.loads(r["data"]) for r in page.get("Items", []))
                if not page.get("LastEvaluatedKey"):
                    break
                args["ExclusiveStartKey"] = page["LastEvaluatedKey"]
        else:
            with self.connect() as db:
                rows = [json.loads(row[0]) for row in db.execute("SELECT data FROM records")]
        return [row for row in rows if row["kind"] == "connection"]

    def get(self, principal, record_id, *, hydrate=True):
        scope = self.scope(principal)
        if self.table is not None:
            item = self.table.get_item(
                Key={"scope": scope, "id": record_id}, ConsistentRead=True
            ).get("Item")
            row = item and item["data"]
        else:
            with self.connect() as db:
                found = db.execute(
                    "SELECT data FROM records WHERE scope=? AND id=?", (scope, record_id)
                ).fetchone()
                row = found and found[0]
        if row is None:
            raise HTTPException(404, "Record not found in your workspace")
        record = json.loads(row)
        if hydrate and record.get("body_key"):
            if self.s3 is None:
                raise HTTPException(503, "Document storage is not configured")
            body = self.s3.get_object(
                Bucket=self.settings.workspace_source_bucket, Key=record["body_key"]
            )["Body"]
            try:
                record.update(json.loads(body.read()))
            finally:
                body.close()
        return record

    def put_blob(self, principal, record_id, data, content_type):
        if self.s3 is None:
            return None
        tenant = hashlib.sha256(self.scope(principal).encode()).hexdigest()
        digest = hashlib.sha256(data).hexdigest()
        key = f"lens/{tenant}/{record_id}/{digest}"
        options = {"ServerSideEncryption": "aws:kms"}
        if self.settings.workspace_kms_key_arn:
            options["SSEKMSKeyId"] = self.settings.workspace_kms_key_arn
        self.s3.put_object(
            Bucket=self.settings.workspace_source_bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            **options,
        )
        return key

    def source_available(self, principal, doc):
        if doc.get("deleted") or doc.get("status") in {"source-error", "index-failed", "indexing"}:
            return False
        if doc.get("source_provider") == "sharepoint":
            from app.workspace_sync import elapsed

            try:
                connection = self.get(principal, doc["connection_id"], hydrate=False)
            except HTTPException as error:
                if error.status_code != 404:
                    raise
                return False
            if connection.get("status") == "failed":
                return False
            if elapsed(connection.get("last_sync")) > 3600:
                return False
        return True

    def save(self, principal, kind, payload, record_id=None, expected_revision=None):
        record_id = record_id or uuid4().hex
        scope = self.scope(principal)
        with self.lock:
            try:
                previous = self.get(principal, record_id, hydrate=False)
            except HTTPException as error:
                if error.status_code != 404:
                    raise
                previous = None
            revision = previous["revision"] if previous else 0
            if expected_revision is not None and revision != expected_revision:
                raise HTTPException(409, "This record changed. Refresh before saving.")
            record = {
                **payload,
                "id": record_id,
                "kind": kind,
                "revision": revision + 1,
                "created_at": previous["created_at"] if previous else now(),
                "updated_at": now(),
                "owner": principal.subject,
                "security_domain": principal.security_domain,
            }
            stored = dict(record)
            if kind == "document" and self.s3 is not None and "parsed" in record:
                body = {k: stored.pop(k) for k in ("text", "parsed") if k in stored}
                stored["body_key"] = self.put_blob(
                    principal, record_id, json.dumps(body).encode(), "application/json"
                )
                record["body_key"] = stored["body_key"]
            encoded = json.dumps(stored, ensure_ascii=False)
            if len(encoded.encode()) > 350_000:
                raise HTTPException(413, "Record exceeds the portable workspace limit (350 KB).")
            if self.table is not None:
                from botocore.exceptions import ClientError

                kwargs = {
                    "Item": {
                        "scope": scope,
                        "id": record_id,
                        "data": encoded,
                        "revision": revision + 1,
                        "kind": kind,
                    },
                    "ConditionExpression": "attribute_not_exists(id)",
                }
                if previous:
                    kwargs.update(
                        ConditionExpression="revision = :previous",
                        ExpressionAttributeValues={":previous": revision},
                    )
                try:
                    self.table.put_item(**kwargs)
                except ClientError as error:
                    if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
                        raise HTTPException(409, "Record changed; refresh and retry.") from error
                    raise
            else:
                with self.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    current = db.execute(
                        "SELECT data FROM records WHERE scope=? AND id=?", (scope, record_id)
                    ).fetchone()
                    actual = json.loads(current[0])["revision"] if current else 0
                    if actual != revision:
                        raise HTTPException(409, "Record changed; refresh and retry.")
                    db.execute(
                        "INSERT OR REPLACE INTO records VALUES (?,?,?)", (scope, record_id, encoded)
                    )
            return record

    def event(self, principal, action, record_id, details):
        # Deliberately excludes document bodies and credentials.
        return self.save(
            principal, "event", {"action": action, "record_id": record_id, "details": details}
        )
