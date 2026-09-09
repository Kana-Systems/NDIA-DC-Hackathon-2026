#!/usr/bin/env python3
"""Check an exported workspace envelope offline, without opening source URLs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.workspace_export import WorkspaceExport, record_digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="JSON downloaded from Reviewed records")
    args = parser.parse_args()
    try:
        if args.file.stat().st_size > 2_000_000:
            raise ValueError("File exceeds the workspace handoff limit")
        envelope = WorkspaceExport.model_validate(json.loads(args.file.read_text("utf-8")))
        if envelope.record_sha256 != record_digest(envelope.record):
            raise ValueError("Record checksum does not match")
        if envelope.record.get("kind") not in {"review", "structured-record"}:
            raise ValueError("Unsupported record type")
        if envelope.record.get("decision") != "approved":
            raise ValueError("Record does not contain an approval")
    except (OSError, ValueError):
        # Validation exceptions may include the full input: keep document content out of logs.
        print(
            "Invalid handoff: check file format, approval, schema version, and checksum.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Valid schema {envelope.schema_version}; record checksum matches; "
        f"{len(envelope.sources)} sources."
    )
    print("Verify current source access and freshness in the originating workspace before reuse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
