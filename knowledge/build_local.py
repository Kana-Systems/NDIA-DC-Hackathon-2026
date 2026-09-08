"""Build a versioned FAR/DFARS text index and reference graph from official DITA."""

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from knowledge.ingest_dita import parse_topic


def build(source_root: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(".building.sqlite")
    connection = sqlite3.connect(staging)
    connection.executescript("""
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS links;
        DROP TABLE IF EXISTS metadata;
        CREATE VIRTUAL TABLE chunks USING fts5(
            evidence_id UNINDEXED, heading, text, authority UNINDEXED,
            url UNINDEXED, version UNINDEXED, document_id UNINDEXED,
            locator UNINDEXED, content_sha256 UNINDEXED);
        CREATE TABLE links(source TEXT, target TEXT, relation TEXT, version TEXT);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
    """)
    retrieved_at = datetime.now(UTC).isoformat()
    stats = {"retrieved_at": retrieved_at, "documents": 0, "chunks": 0, "links": 0, "sources": {}}
    for authority in ("FAR", "DFARS"):
        root = source_root / authority.lower()
        version = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
        stats["sources"][authority] = {
            "repository": f"https://github.com/GSA/GSA-Acquisition-{authority}",
            "commit": version,
        }
        for path in sorted((root / "dita").glob("*.dita")):
            records = parse_topic(
                path,
                authority=authority,
                document_id=f"{authority}:{path.stem}",
                document_title=f"{authority} {path.stem}",
            )
            if not records:
                continue
            stats["documents"] += 1
            url = f"https://github.com/GSA/GSA-Acquisition-{authority}/blob/{version}/dita/{path.name}"
            for record_index, record in enumerate(records):
                # Overlap preserves references crossing a chunk boundary.
                for offset in range(0, len(record.text), 1400):
                    chunk = record.text[offset : offset + 1800]
                    if not chunk.strip():
                        continue
                    digest = hashlib.sha256(chunk.encode()).hexdigest()
                    evidence_id = f"{authority}:{path.stem}:{record_index}:{offset}:{digest[:12]}"
                    connection.execute(
                        "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            evidence_id,
                            record.heading,
                            chunk,
                            authority,
                            url,
                            version,
                            f"{authority}:{path.stem}",
                            record.locator,
                            digest,
                        ),
                    )
                    stats["chunks"] += 1
            targets = set()
            for element in ElementTree.parse(path).getroot().iter():
                href = element.get("href", "")
                if ".dita" in href:
                    target = Path(href.split("#", 1)[0]).stem
                    if re.fullmatch(r"[\w.-]+", target):
                        targets.add(target)
            for target in sorted(targets):
                connection.execute(
                    "INSERT INTO links VALUES(?,?,?,?)",
                    (
                        f"{authority}:{path.stem}",
                        f"{authority}:{target}",
                        "references",
                        version,
                    ),
                )
                stats["links"] += 1
    if not stats["chunks"]:
        raise ValueError("Official source extraction produced no chunks")
    connection.execute("INSERT INTO metadata VALUES(?,?)", ("manifest", json.dumps(stats)))
    connection.commit()
    connection.close()
    staging.replace(destination)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=Path("artifacts/sources"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/knowledge/federal.sqlite"))
    args = parser.parse_args()
    print(json.dumps(build(args.sources, args.output), indent=2))
