"""Canonical source-record schema and deterministic identifiers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: str, default: str = "unknown") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _clean(value).lower()).strip("-")
    return slug or default


def make_citation_id(authority: str, document_id: str, locator: str) -> str:
    """Return an ID stable across text/metadata corrections.

    The readable prefix aids debugging while the digest prevents collisions.
    Content is intentionally excluded so citations survive source updates.
    """

    identity = "|".join(_clean(v).casefold() for v in (authority, document_id, locator))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{_slug(authority)}:{_slug(document_id)}:{_slug(locator)}:{digest}"


@dataclass(frozen=True)
class SourceRecord:
    source_type: str
    authority: str
    document_title: str
    document_id: str
    locator: str
    heading: str
    text: str
    url: str = ""
    effective_date: str = ""
    retrieved_at: str = ""
    corpus_id: str = "government-contracts"
    source_document_id: str = ""
    version: str = "1"
    security_label: str = "public"
    acl_principals: tuple[str, ...] = ("public",)
    entity_ids: tuple[str, ...] = ()
    parent_citation_id: str = ""
    ingested_at: str = ""
    deleted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    citation_id: str = ""
    schema_version: str = SCHEMA_VERSION
    content_sha256: str = ""

    def normalized(self) -> SourceRecord:
        values = {
            name: _clean(getattr(self, name))
            for name in (
                "source_type",
                "authority",
                "document_title",
                "document_id",
                "locator",
                "heading",
                "text",
                "url",
                "effective_date",
                "retrieved_at",
                "corpus_id",
                "source_document_id",
                "version",
                "security_label",
                "parent_citation_id",
                "ingested_at",
            )
        }
        if not values["text"]:
            raise ValueError("source record text cannot be empty")
        if not values["document_id"] or not values["locator"]:
            raise ValueError("document_id and locator are required")
        if not values["source_document_id"]:
            values["source_document_id"] = values["document_id"]
        citation_id = self.citation_id or make_citation_id(
            values["authority"], values["document_id"], values["locator"]
        )
        content_sha256 = hashlib.sha256(values["text"].encode("utf-8")).hexdigest()
        metadata = json.loads(json.dumps(self.metadata, sort_keys=True, default=str))
        acl_principals = tuple(
            dict.fromkeys(_clean(value) for value in self.acl_principals if _clean(value))
        )
        if not acl_principals:
            raise ValueError("acl_principals cannot be empty")
        entity_ids = tuple(
            dict.fromkeys(_clean(value) for value in self.entity_ids if _clean(value))
        )
        return SourceRecord(
            **values,
            acl_principals=acl_principals,
            entity_ids=entity_ids,
            deleted=bool(self.deleted),
            metadata=metadata,
            citation_id=citation_id,
            schema_version=SCHEMA_VERSION,
            content_sha256=content_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


def write_jsonl(records: Iterable[SourceRecord], destination: str) -> int:
    count = 0
    with open(destination, "w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(source: str) -> list[dict[str, Any]]:
    with open(source, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
