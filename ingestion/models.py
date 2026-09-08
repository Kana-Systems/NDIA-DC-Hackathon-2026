"""Transport-neutral models for enterprise document ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


def canonical_checksum(text: str) -> str:
    """Hash UTF-8 source text without exposing it to logs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChangeKind(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


@dataclass(frozen=True)
class SourceDocument:
    """A document emitted by a connector before deterministic chunking."""

    corpus: str
    document_id: str
    version: str
    text: str
    security_label: str = "public"
    acl_principals: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

    def normalized(self) -> SourceDocument:
        corpus = self.corpus.strip()
        document_id = self.document_id.strip()
        version = str(self.version).strip()
        security_label = self.security_label.strip().casefold()
        if not corpus or not document_id or not version:
            raise ValueError("corpus, document_id, and version are required")
        if not self.text.strip():
            raise ValueError("source document text cannot be empty")
        checksum = canonical_checksum(self.text)
        if self.checksum and self.checksum != checksum:
            raise ValueError("source document checksum does not match its text")
        provenance = json.loads(json.dumps(self.provenance, sort_keys=True, default=str))
        principals = tuple(sorted({item.strip() for item in self.acl_principals if item.strip()}))
        return SourceDocument(
            corpus=corpus,
            document_id=document_id,
            version=version,
            text=self.text,
            security_label=security_label,
            acl_principals=principals,
            provenance=provenance,
            checksum=checksum,
        )


@dataclass(frozen=True)
class ChangeEvent:
    kind: ChangeKind
    document: SourceDocument | None = None
    corpus: str = ""
    document_id: str = ""
    version: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def upsert(cls, document: SourceDocument) -> ChangeEvent:
        return cls(ChangeKind.UPSERT, document=document.normalized())

    @classmethod
    def delete(
        cls,
        corpus: str,
        document_id: str,
        *,
        version: str = "",
        provenance: dict[str, Any] | None = None,
    ) -> ChangeEvent:
        if not corpus.strip() or not document_id.strip():
            raise ValueError("corpus and document_id are required for a tombstone")
        return cls(
            ChangeKind.DELETE,
            corpus=corpus.strip(),
            document_id=document_id.strip(),
            version=str(version).strip(),
            provenance=provenance or {},
        )

    @property
    def key(self) -> tuple[str, str]:
        if self.document is not None:
            return self.document.corpus, self.document.document_id
        return self.corpus, self.document_id


@dataclass(frozen=True)
class ConnectorBatch:
    events: tuple[ChangeEvent, ...]
    cursor: str


@dataclass(frozen=True)
class NormalizedRecord:
    """A retrieval-ready chunk retaining mandatory authorization metadata."""

    record_id: str
    corpus: str
    document_id: str
    version: str
    chunk_index: int
    text: str
    security_label: str
    acl_principals: tuple[str, ...]
    provenance: dict[str, Any]
    checksum: str
    document_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_index_dict(self) -> dict[str, Any]:
        """Map a connector chunk to the strict enterprise OpenSearch schema."""

        return {
            "citation_id": self.record_id,
            "schema_version": "1.0",
            "source_type": str(self.provenance.get("connector") or "connector"),
            "authority": str(self.provenance.get("authority") or "Approved source"),
            "document_title": str(self.provenance.get("name") or self.document_id),
            "document_id": self.document_id,
            "locator": f"chunk-{self.chunk_index + 1}",
            "heading": str(self.provenance.get("heading") or self.document_id),
            "text": self.text,
            "content_sha256": self.checksum,
            "effective_date": str(self.provenance.get("effective_date") or ""),
            "retrieved_at": str(self.provenance.get("retrieved_at") or ""),
            "url": str(self.provenance.get("web_url") or ""),
            "corpus_id": self.corpus,
            "source_document_id": self.document_id,
            "version": self.version,
            "security_label": self.security_label,
            "acl_principals": list(self.acl_principals),
            "entity_ids": list(self.provenance.get("entity_ids") or []),
            "parent_citation_id": "",
            "ingested_at": str(self.provenance.get("ingested_at") or ""),
            "deleted": False,
            "metadata": {
                key: value
                for key, value in self.provenance.items()
                if key not in {"name", "web_url"}
            },
        }
