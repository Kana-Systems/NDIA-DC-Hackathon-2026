"""Create and populate an AWS OpenSearch index using SigV4 authentication."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from knowledge.opensearch_bulk import build_bulk, index_definition
from knowledge.schema import read_jsonl

DEFAULT_DIMENSIONS = 1024
DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"


class OpenSearchRequestError(RuntimeError):
    """Sanitized OpenSearch transport error."""

    def __init__(self, method: str, path: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"OpenSearch {method} {path} failed with HTTP {status_code}")


def resolve_service(endpoint: str, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    return "aoss" if ".aoss." in endpoint.casefold() else "es"


def titan_embedder(
    region: str,
    model_id: str = DEFAULT_MODEL_ID,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> Callable[[str], list[float]]:
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("Titan embeddings require boto3") from error
    client = boto3.client("bedrock-runtime", region_name=region)

    def embed(text: str) -> list[float]:
        request = json.dumps({"inputText": text, "dimensions": dimensions, "normalize": True})
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=request,
        )
        payload = json.loads(response["body"].read())
        return [float(value) for value in payload["embedding"]]

    return embed


def prepare_records(
    records: list[dict[str, Any]],
    embedder: Callable[[str], list[float]] | None = None,
    dimensions: int = DEFAULT_DIMENSIONS,
) -> list[dict[str, Any]]:
    prepared = []
    for original in records:
        if not original.get("citation_id") or not original.get("text"):
            raise ValueError("every record requires citation_id and text")
        record = dict(original)
        if embedder:
            embedding = embedder(record["text"])
            if len(embedding) != dimensions:
                raise ValueError(
                    f"embedding for {record['citation_id']} has {len(embedding)} "
                    f"dimensions; expected {dimensions}"
                )
            record["embedding"] = embedding
        prepared.append(record)
    return prepared


class SigV4OpenSearchClient:
    def __init__(self, endpoint: str, region: str, service: str = "auto") -> None:
        if not endpoint.startswith("https://"):
            raise ValueError("OpenSearch endpoint must use https://")
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError as error:
            raise RuntimeError("AWS indexing requires boto3 and botocore") from error
        self.endpoint = endpoint.rstrip("/")
        self.region = region
        self.service = resolve_service(endpoint, service)
        self.credentials = boto3.Session().get_credentials()
        if self.credentials is None:
            raise RuntimeError("AWS credentials were not found")
        self._request_type = AWSRequest
        self._signer_type = SigV4Auth

    def request(
        self,
        method: str,
        path: str,
        body: str = "",
        content_type: str = "",
    ) -> dict[str, Any]:
        url = f"{self.endpoint}/{path.lstrip('/')}"
        data = body.encode("utf-8") if body else None
        headers = {"Content-Type": content_type} if content_type else {}
        aws_request = self._request_type(
            method=method,
            url=url,
            data=data,
            headers=headers,
        )
        credentials = self.credentials.get_frozen_credentials()
        self._signer_type(credentials, self.service, self.region).add_auth(aws_request)
        prepared = aws_request.prepare()
        request = urllib.request.Request(
            url,
            data=data,
            headers=dict(prepared.headers.items()),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_body = response.read()
                return json.loads(response_body) if response_body else {}
        except urllib.error.HTTPError as error:
            status_code = error.code
            error.close()
            raise OpenSearchRequestError(method, path, status_code) from None


def index_exists(client: SigV4OpenSearchClient, encoded_index: str) -> bool:
    try:
        client.request("HEAD", encoded_index)
    except OpenSearchRequestError as error:
        if error.status_code == 404:
            return False
        raise
    return True


def ensure_index(
    client: SigV4OpenSearchClient,
    encoded_index: str,
    dimensions: int | None,
) -> bool:
    """Create an index only when absent, tolerating a concurrent creator."""

    if index_exists(client, encoded_index):
        return False
    mapping = json.dumps(index_definition(dimensions), sort_keys=True)
    try:
        client.request("PUT", encoded_index, mapping, "application/json")
    except OpenSearchRequestError as error:
        if error.status_code not in {400, 409} or not index_exists(client, encoded_index):
            raise
        return False
    return True


def index_records(
    client: SigV4OpenSearchClient,
    records: list[dict[str, Any]],
    index: str,
    dimensions: int | None,
    create_index: bool = True,
) -> dict[str, Any]:
    encoded_index = urllib.parse.quote(index, safe="-_")
    if create_index:
        ensure_index(client, encoded_index, dimensions)
    result = client.request(
        "POST",
        "_bulk?refresh=wait_for",
        build_bulk(records, index),
        "application/x-ndjson",
    )
    if result.get("errors"):
        failed = [
            item for item in result.get("items", []) if next(iter(item.values())).get("error")
        ]
        raise RuntimeError(f"OpenSearch bulk indexing reported {len(failed)} failures")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", help="Normalized JSONL records")
    parser.add_argument("--endpoint")
    parser.add_argument("--index", default="government-contract-knowledge-v1")
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-gov-west-1"))
    parser.add_argument("--service", choices=("auto", "es", "aoss"), default="auto")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--dimensions", type=int, choices=(256, 512, 1024), default=1024)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--bedrock-region")
    parser.add_argument("--skip-create", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bulk-output", type=Path)
    parser.add_argument("--mapping-output", type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.records)
    embed = None
    if args.embed and not args.dry_run:
        embed = titan_embedder(
            args.bedrock_region or args.region,
            args.model_id,
            args.dimensions,
        )
    prepared = prepare_records(records, embed, args.dimensions)
    vector_dimensions = args.dimensions if args.embed else None
    if args.bulk_output:
        args.bulk_output.write_text(
            build_bulk(prepared, args.index),
            encoding="utf-8",
        )
    if args.mapping_output:
        mapping = index_definition(vector_dimensions)
        args.mapping_output.write_text(
            json.dumps(mapping, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "records": len(prepared),
                    "index": args.index,
                    "region": args.region,
                    "service": resolve_service(args.endpoint or "", args.service),
                    "embedding_requests": 0,
                    "embedding_dimensions": vector_dimensions,
                },
                sort_keys=True,
            )
        )
        return
    if not args.endpoint:
        parser.error("--endpoint is required unless --dry-run is used")
    client = SigV4OpenSearchClient(args.endpoint, args.region, args.service)
    result = index_records(
        client,
        prepared,
        args.index,
        vector_dimensions,
        not args.skip_create,
    )
    print(json.dumps({"indexed": len(prepared), "took_ms": result.get("took")}))


if __name__ == "__main__":
    main()
