"""Scheduled Microsoft Graph delta ingestion entry point."""

from __future__ import annotations

import json
import os

import boto3

from ingestion.aws_store import AwsIngestionStore
from ingestion.graph import GraphClient, GraphDeltaConnector
from ingestion.pipeline import ingest


def main() -> int:
    region = os.getenv("AWS_REGION", "us-gov-west-1")
    secret_arn = os.environ["GRAPH_CONNECTOR_SECRET_ARN"]
    secret_value = boto3.client(
        "secretsmanager",
        region_name=region,
    ).get_secret_value(SecretId=secret_arn)["SecretString"]
    configuration = json.loads(secret_value)
    required = ("client_id", "client_secret", "tenant_id", "drive_id")
    missing = [name for name in required if not configuration.get(name)]
    if missing:
        raise ValueError(f"Graph connector secret is missing required fields: {', '.join(missing)}")
    store = AwsIngestionStore.from_environment()
    cursor_id = f"connector#graph#{configuration['drive_id']}"
    cursor_item = store.document_table.get_item(
        Key={"document_id": cursor_id},
        ConsistentRead=True,
    ).get("Item", {})
    client = GraphClient(
        client_id=configuration["client_id"],
        client_secret=configuration["client_secret"],
        tenant_id=configuration["tenant_id"],
        graph_base_url=configuration.get(
            "graph_base_url",
            "https://graph.microsoft.us/v1.0",
        ),
        token_url=configuration.get("token_url"),
    )
    connector = GraphDeltaConnector(
        client,
        drive_id=configuration["drive_id"],
        corpus=configuration.get("corpus", "j2-sharepoint"),
        security_label=configuration.get("security_label", "demo"),
        default_acl_principals=tuple(configuration.get("default_acl_principals", [])),
    )
    result = ingest(connector, store, cursor=cursor_item.get("cursor"))
    store.document_table.put_item(
        Item={
            "document_id": cursor_id,
            "record_type": "connector_cursor",
            "cursor": result.cursor,
            "deleted": True,
        }
    )
    print(
        json.dumps(
            {
                "connector": "microsoft-graph",
                "changes_received": result.changes,
                "inserted": result.applied.inserted,
                "updated": result.applied.updated,
                "deleted": result.applied.deleted,
                "unchanged": result.applied.unchanged,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
