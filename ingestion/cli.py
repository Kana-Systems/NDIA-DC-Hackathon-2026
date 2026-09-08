"""CLI for automatically generating and ingesting the public fixture corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ingestion.aws_store import AwsIngestionStore
from ingestion.connectors import FilesystemConnector
from ingestion.fixtures import FIXTURE_DOCUMENT_COUNT, generate_fixture_corpus
from ingestion.pipeline import ingest
from ingestion.store import InMemoryIngestionStore


def default_fixture_path() -> Path:
    return Path(os.getenv("FIXTURE_SOURCE", "/tmp/j2-fixtures"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_fixture_path())
    parser.add_argument("--count", type=int, default=FIXTURE_DOCUMENT_COUNT)
    parser.add_argument("--max-chars", type=int, default=1_200)
    parser.add_argument(
        "--durable",
        action="store_true",
        default=os.getenv("DURABLE_INGESTION_ENABLED", "false").lower() == "true",
    )
    args = parser.parse_args(argv)

    generated = generate_fixture_corpus(args.source, count=args.count)
    connector = FilesystemConnector(
        args.source,
        corpus="j2-public-synthetic",
        security_label="public",
        acl_principals=("public",),
    )
    store = (
        AwsIngestionStore.from_environment(max_chunk_chars=args.max_chars)
        if args.durable
        else InMemoryIngestionStore(max_chunk_chars=args.max_chars)
    )
    result = ingest(connector, store)
    processed = result.applied.inserted + result.applied.updated + result.applied.unchanged
    records_stored = (
        store.document_count() if isinstance(store, AwsIngestionStore) else len(store.records)
    )
    print(
        json.dumps(
            {
                "documents_generated": len(generated),
                "documents_ingested": processed,
                "documents_inserted": result.applied.inserted,
                "documents_updated": result.applied.updated,
                "documents_unchanged": result.applied.unchanged,
                "records_stored": records_stored,
                "durable": isinstance(store, AwsIngestionStore),
                "security_label": "public",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
