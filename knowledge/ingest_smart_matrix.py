"""Ingest DAU/DPAP Smart Matrix CSV or XLSX exports."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from knowledge.schema import SourceRecord, write_jsonl

ALIASES = {
    "number": ("clause number", "provision/clause", "number", "far/dfars"),
    "title": ("title", "clause title", "provision or clause title"),
    "prescription": ("prescription", "prescribed in"),
    "applicability": ("applicability", "conditions", "use when"),
    "type": ("type", "provision/clause type"),
    "url": ("url", "link"),
}


def _normalized_row(row: dict[str, Any]) -> dict[str, str]:
    lowered = {
        str(key or "").strip().casefold(): str(value or "").strip() for key, value in row.items()
    }
    result: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        result[target] = next((lowered[name] for name in aliases if lowered.get(name)), "")
    return result


def read_rows(path: Path, sheet: str | None = None) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
        return
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("Smart Matrix input must be CSV, XLSX, or XLSM")
    try:
        import openpyxl
    except ImportError as error:
        raise RuntimeError("XLSX ingestion requires: pip install openpyxl") from error
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[sheet] if sheet else workbook.active
    rows = worksheet.iter_rows(values_only=True)
    headers = [str(value or "").strip() for value in next(rows)]
    for values in rows:
        yield dict(zip(headers, values, strict=True))


def ingest(path: Path, *, sheet: str | None = None, retrieved_at: str = "") -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for row_number, raw in enumerate(read_rows(path, sheet), start=2):
        row = _normalized_row(raw)
        if not row["number"]:
            continue
        text_parts = [
            value
            for value in (
                f"Applicability: {row['applicability']}" if row["applicability"] else "",
                f"Prescription: {row['prescription']}" if row["prescription"] else "",
                f"Type: {row['type']}" if row["type"] else "",
            )
            if value
        ]
        records.append(
            SourceRecord(
                source_type="smart_matrix",
                authority="DAU Smart Matrix",
                document_title="Provision and Clause Matrix",
                document_id="smart-matrix",
                locator=row["number"],
                heading=row["title"] or row["number"],
                text=" ".join(text_parts) or f"Matrix entry for {row['number']}.",
                url=row["url"],
                retrieved_at=retrieved_at,
                metadata={"row_number": row_number, "clause_number": row["number"]},
            ).normalized()
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output")
    parser.add_argument("--sheet")
    parser.add_argument("--retrieved-at", default="")
    args = parser.parse_args()
    records = ingest(
        args.source,
        sheet=args.sheet,
        retrieved_at=args.retrieved_at,
    )
    count = write_jsonl(records, args.output)
    print(f"wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
