"""Owner-filtered lexical/vector retrieval with authoritative version rechecks."""

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from app.models import Evidence
from knowledge.index_opensearch import (
    OpenSearchRequestError,
    SigV4OpenSearchClient,
    titan_embedder,
)


class WorkspaceSearch:
    def __init__(self, settings, client=None, embed=None):
        self.settings = settings
        self.index = settings.workspace_search_index
        if not self.index or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.index
        ):
            raise ValueError("Invalid workspace search index")
        self.client = client or SigV4OpenSearchClient(
            settings.opensearch_endpoint or "", settings.aws_region
        )
        self.embed = embed or titan_embedder(settings.aws_region, settings.titan_embedding_model_id)

    def ensure_index(self, index=None):
        index = index or self.index
        try:
            self.client.request("HEAD", index)
            return
        except OpenSearchRequestError as error:
            if error.status_code != 404:
                raise
        mapping = {
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    **{
                        key: {"type": "keyword"}
                        for key in (
                            "owner",
                            "security_domain",
                            "document_id",
                            "version",
                            "category",
                            "evidence_id",
                        )
                    },
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "url": {"type": "keyword", "index": False},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {"name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"},
                    },
                },
            },
        }
        try:
            self.client.request("PUT", index, json.dumps(mapping), "application/json")
        except OpenSearchRequestError as error:
            if error.status_code not in {400, 409}:
                raise
            self.client.request("HEAD", index)

    def federal_index(self):
        """A snapshot-specific index prevents mixing superseded regulatory versions."""
        from app.local_retrieval import FederalCorpusRetrieval

        manifest = FederalCorpusRetrieval(self.settings.local_corpus_path).manifest()
        version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
        return f"{self.index}-federal-{version}", manifest

    def index_federal(self):
        index, manifest = self.federal_index()
        self.ensure_index(index)
        count = self.client.request("GET", f"{index}/_count")["count"]
        if count == manifest["chunks"]:
            return count
        path = Path(self.settings.local_corpus_path).resolve().as_uri()
        with closing(sqlite3.connect(f"{path}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            cursor = db.execute("SELECT * FROM chunks")
            while rows := cursor.fetchmany(100):
                lines = []
                for row in rows:
                    lines.extend(
                        (
                            json.dumps({"index": {"_index": index, "_id": row["evidence_id"]}}),
                            json.dumps(
                                {
                                    "owner": "official",
                                    "security_domain": "public",
                                    "document_id": row["document_id"],
                                    "version": row["version"],
                                    "category": "regulation",
                                    "evidence_id": row["evidence_id"],
                                    "title": row["heading"],
                                    "text": row["text"],
                                    "url": row["url"],
                                }
                            ),
                        )
                    )
                result = self.client.request(
                    "POST", "_bulk", "\n".join(lines) + "\n", "application/x-ndjson"
                )
                if result.get("errors"):
                    raise RuntimeError("Official source indexing failed")
        self.client.request("POST", f"{index}/_refresh")
        count = self.client.request("GET", f"{index}/_count")["count"]
        if count != manifest["chunks"]:
            raise RuntimeError("Official corpus index is incomplete")
        return count

    def retrieve_federal(self, queries):
        index, _ = self.federal_index()
        result = self.client.request(
            "POST",
            f"{index}/_search",
            json.dumps(
                {
                    "size": 4,
                    "_source": ["evidence_id"],
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"owner": "official"}},
                                {"term": {"security_domain": "public"}},
                            ],
                            "must": [
                                {
                                    "multi_match": {
                                        "query": " ".join(queries)[:6000],
                                        "fields": ["title^4", "text"],
                                    }
                                }
                            ],
                        },
                    },
                }
            ),
            "application/json",
        )
        # OpenSearch determines rank; the packaged official snapshot verifies the source text.
        path = Path(self.settings.local_corpus_path).resolve().as_uri()
        evidence = []
        with closing(sqlite3.connect(f"{path}?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            for hit in result.get("hits", {}).get("hits", []):
                row = db.execute(
                    "SELECT * FROM chunks WHERE evidence_id=?", (hit["_source"]["evidence_id"],)
                ).fetchone()
                if row is None:
                    continue
                evidence.append(
                    Evidence(
                        evidence_id=row["evidence_id"],
                        document_id=row["document_id"],
                        version=row["version"],
                        source=f"{row['authority']} {row['locator']}",
                        title=row["heading"],
                        excerpt=row["text"],
                        url=row["url"],
                    )
                )
        return evidence

    def index_document(self, doc):
        self.ensure_index()
        chunks = [doc["text"][start : start + 1200] for start in range(0, len(doc["text"]), 1200)]

        def entry(pair):
            index, text = pair
            evidence_id = f"workspace:{doc['id']}:{doc['version']}:{index}"
            embedding = self.embed(text)
            if len(embedding) != 1024:
                raise ValueError("Embedding dimensions do not match workspace index")
            key = hashlib.sha256(
                f"{doc['security_domain']}:{doc['owner']}:{evidence_id}".encode()
            ).hexdigest()
            return (
                json.dumps({"index": {"_index": self.index, "_id": key}}),
                json.dumps(
                    {
                        "owner": doc["owner"],
                        "security_domain": doc["security_domain"],
                        "document_id": doc["id"],
                        "version": doc["version"],
                        "category": doc["category"],
                        "evidence_id": evidence_id,
                        "title": doc["title"],
                        "text": text,
                        "url": doc.get("source_url", ""),
                        "embedding": embedding,
                    }
                ),
            )

        # Bounded concurrency and bulk payloads support complete documents in S3.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(chunks), 25):
                batch = list(enumerate(chunks[start : start + 25], start))
                lines = [line for pair in pool.map(entry, batch) for line in pair]
                result = self.client.request(
                    "POST",
                    "_bulk?refresh=wait_for",
                    "\n".join(lines) + "\n",
                    "application/x-ndjson",
                )
                if result.get("errors"):
                    raise RuntimeError("Workspace indexing failed; retry indexing")

    def retrieve(self, queries, principal, db, document_id=None, references_only=False):
        filters = [
            {"term": {"owner": principal.subject}},
            {"term": {"security_domain": principal.security_domain}},
        ]
        if references_only:
            filters.append({"bool": {"must_not": [{"term": {"category": "contract"}}]}})
        elif document_id:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"document_id": document_id}},
                            {"bool": {"must_not": [{"term": {"category": "contract"}}]}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        text = " ".join(queries)[:6000]
        self.ensure_index()
        q_filter = {"bool": {"filter": filters}}
        searches = [
            {
                "bool": {
                    "filter": filters,
                    "must": [
                        {
                            "multi_match": {
                                "query": text,
                                "fields": ["title^2", "text"],
                            }
                        }
                    ],
                }
            }
        ]
        if self.settings.opensearch_vector_enabled:
            searches.append(
                {"knn": {"embedding": {"vector": self.embed(text), "k": 20, "filter": q_filter}}}
            )
        ranked, sources = {}, {}
        for query in searches:
            result = self.client.request(
                "POST",
                f"{self.index}/_search",
                json.dumps(
                    {
                        "size": 20,
                        "_source": {"excludes": ["embedding"]},
                        "query": query,
                    }
                ),
                "application/json",
            )
            for rank, hit in enumerate(result.get("hits", {}).get("hits", []), 1):
                source = hit["_source"]
                key = source["evidence_id"]
                sources[key] = source
                ranked[key] = ranked.get(key, 0) + 1 / (60 + rank)
        evidence = []
        from fastapi import HTTPException

        for key in sorted(ranked, key=ranked.get, reverse=True):
            source = sources[key]
            if (
                source["owner"] != principal.subject
                or source["security_domain"] != principal.security_domain
            ):
                continue
            try:
                current = db.get(principal, source["document_id"], hydrate=False)
            except HTTPException as error:
                if error.status_code == 404:
                    continue
                raise
            if (
                not db.source_available(principal, current)
                or current["version"] != source["version"]
            ):
                continue
            if references_only and current["category"] == "contract":
                continue
            if document_id and current["id"] != document_id and current["category"] == "contract":
                continue
            evidence.append(
                Evidence(
                    evidence_id=key,
                    document_id=current["id"],
                    version=current["version"],
                    title=current["title"],
                    source=current["category"],
                    excerpt=source["text"],
                    url=current.get("source_url") or None,
                    security_label=principal.security_domain,
                    acl_principals=[f"user:{principal.subject}"],
                )
            )
            if len(evidence) == 4:
                break
        return evidence
