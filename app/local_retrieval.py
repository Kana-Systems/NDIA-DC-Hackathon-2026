"""Retrieval from actual versioned official source chunks, without seeded quotes."""

import json
import re
import sqlite3
from pathlib import Path

from app.models import Evidence


class FederalCorpusRetrieval:
    def __init__(self, path: str, top_k: int = 6):
        self.path = Path(path).resolve()
        self.top_k = top_k
        if not self.path.is_file():
            raise RuntimeError("Federal corpus is missing; run knowledge.build_local first")

    def connect(self):
        return sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)

    def manifest(self) -> dict:
        with self.connect() as connection:
            return json.loads(
                connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
            )

    def retrieve(self, queries, principal=None, filters=None) -> list[Evidence]:
        ranked = {}
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            for query in queries:
                terms = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", query)))[:24]
                if not terms:
                    continue
                expression = " OR ".join('"' + term + '"' for term in terms)
                rows = connection.execute(
                    "SELECT *, bm25(chunks, 0, 4, 1) AS rank FROM chunks "
                    "WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
                    (expression, self.top_k),
                )
                for row in rows:
                    if (
                        filters
                        and filters.document_ids
                        and row["document_id"] not in filters.document_ids
                    ):
                        continue
                    item = dict(row)
                    exact_references = re.findall(r"(?:252|52)\.\d{3}-\d+", query)
                    if any(
                        re.search(re.escape(ref) + r"(?!\d)", item["heading"])
                        for ref in exact_references
                    ):
                        item["rank"] -= 100
                    previous = ranked.get(item["evidence_id"])
                    if previous is None or item["rank"] < previous["rank"]:
                        ranked[item["evidence_id"]] = item
        best = sorted(ranked.values(), key=lambda item: item["rank"])[: self.top_k]
        return [
            Evidence(
                evidence_id=item["evidence_id"],
                source=f"{item['authority']} {item['locator']}",
                title=item["heading"],
                excerpt=item["text"],
                url=item["url"],
                document_id=item["document_id"],
                version=item["version"],
            )
            for item in best
        ]

    def relations(self, document_ids: list[str]) -> list[dict]:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for document_id in dict.fromkeys(document_ids)
                for row in connection.execute(
                    "SELECT * FROM links WHERE source=? LIMIT 20",
                    (document_id,),
                )
            ]
