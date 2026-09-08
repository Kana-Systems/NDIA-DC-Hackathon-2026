"""Validate human-reviewed federal labels before allowing supervised training."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from ml.preprocess_cuad import write_jsonl
from ml.tune_models import atomic_json, digest


def validate(records, known_labels):
    ids, families, texts = {}, {}, {}
    if not records:
        raise ValueError("No reviewed federal examples supplied")
    for row in records:
        if row.get("review_status") != "approved" or row.get("reviewer_type") != "human":
            raise ValueError("Draft or machine-generated labels cannot enter supervised training")
        if not row.get("reviewed_by", "").strip() or not row.get("review_notes", "").strip():
            raise ValueError("Reviewer identity and rationale are required")
        datetime.fromisoformat(row["reviewed_at"])
        if row.get("synthetic") is not False:
            raise ValueError("This importer requires non-synthetic federal examples")
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("An explicit frozen split is required")
        if (
            not row.get("text", "").strip()
            or not row.get("source_urls")
            or not row.get("source_version")
        ):
            raise ValueError("Text and source provenance are required")
        if any(not url.startswith("https://") for url in row["source_urls"]):
            raise ValueError("Source URLs must use HTTPS")
        if not isinstance(row.get("labels"), list) or set(row["labels"]) - set(known_labels):
            raise ValueError("Unknown or malformed labels")
        if not row.get("document_id") or not row.get("family_id"):
            raise ValueError("Document and contract-family IDs are required")
        text_hash = hashlib.sha256(" ".join(row["text"].lower().split()).encode()).hexdigest()
        for mapping, key in (
            (ids, row["document_id"]),
            (families, row["family_id"]),
            (texts, text_hash),
        ):
            previous = mapping.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Document, template-family or exact-text split leakage")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    records = validate(json.loads(args.source.read_text()), labels)
    # This writes a staging set, not the active CUAD data or a selected model.
    for split in ("train", "validation", "test"):
        write_jsonl(
            args.output / f"{split}.jsonl", [row for row in records if row["split"] == split]
        )
    atomic_json(args.output / "labels.json", labels)
    atomic_json(
        args.output / "provenance.json",
        {
            "source_sha256": digest(args.source),
            "labels_sha256": digest(args.labels),
            "examples": len(records),
            "reviewer_assertions_validated": True,
            "reviewer_credentials_independently_verified": False,
            "next": "Review permissions, taxonomy, windowing and data quality before training",
        },
    )


if __name__ == "__main__":
    main()
