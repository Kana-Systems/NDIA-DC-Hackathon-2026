"""Create reproducible OpenSearch mappings and Bulk API payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge.schema import read_jsonl


def index_definition(vector_dimensions: int | None = None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "citation_id": {"type": "keyword"},
        "schema_version": {"type": "keyword"},
        "source_type": {"type": "keyword"},
        "authority": {"type": "keyword"},
        "document_title": {"type": "text"},
        "document_id": {"type": "keyword"},
        "locator": {"type": "keyword"},
        "heading": {"type": "text"},
        "text": {"type": "text", "similarity": "BM25"},
        "content_sha256": {"type": "keyword"},
        "effective_date": {"type": "date", "ignore_malformed": True},
        "retrieved_at": {"type": "date", "ignore_malformed": True},
        "url": {"type": "keyword", "index": False},
        "corpus_id": {"type": "keyword"},
        "source_document_id": {"type": "keyword"},
        "version": {"type": "keyword"},
        "security_label": {"type": "keyword"},
        "acl_principals": {"type": "keyword"},
        "entity_ids": {"type": "keyword"},
        "parent_citation_id": {"type": "keyword"},
        "ingested_at": {"type": "date", "ignore_malformed": True},
        "deleted": {"type": "boolean"},
        "metadata": {"type": "object", "enabled": False},
    }
    settings: dict[str, Any] = {"index": {"number_of_shards": 1}}
    if vector_dimensions:
        settings["index"]["knn"] = True
        properties["embedding"] = {
            "type": "knn_vector",
            "dimension": vector_dimensions,
            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
        }
    return {"settings": settings, "mappings": {"dynamic": "strict", "properties": properties}}


def build_bulk(records: list[dict[str, Any]], index: str) -> str:
    lines: list[str] = []
    for record in records:
        citation_id = record["citation_id"]
        lines.append(json.dumps({"index": {"_index": index, "_id": citation_id}}, sort_keys=True))
        lines.append(json.dumps(record, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare payloads; does not contact OpenSearch.")
    parser.add_argument("records", help="Normalized JSONL records")
    parser.add_argument("bulk_output", type=Path)
    parser.add_argument("--index", default="government-contract-knowledge-v1")
    parser.add_argument("--vector-dimensions", type=int)
    parser.add_argument("--mapping-output", type=Path)
    args = parser.parse_args()
    records = read_jsonl(args.records)
    args.bulk_output.write_text(build_bulk(records, args.index), encoding="utf-8")
    if args.mapping_output:
        args.mapping_output.write_text(
            json.dumps(index_definition(args.vector_dimensions), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"prepared {len(records)} bulk index operations")


if __name__ == "__main__":
    main()
