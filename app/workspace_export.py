"""Portable, versioned handoff contract for analyst-reviewed workspace outputs."""

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.workspace_store import now


class SourceManifestEntry(BaseModel):
    document_id: str
    version: str
    title: str
    source: str
    security_label: str
    evidence_ids: list[str] = Field(default_factory=list)
    role: Literal["context", "evidence", "context-and-evidence"] = "evidence"


class WorkspaceExport(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    exported_at: str
    external_write_performed: Literal[False] = False
    disclaimer: str = "Analyst-reviewed screening output; not a compliance certification."
    record_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    hash_format: Literal["sha256:json-sort-keys-utf8-v1"] = "sha256:json-sort-keys-utf8-v1"
    sources: list[SourceManifestEntry]
    record: dict[str, Any]


def record_digest(record):
    """Digest the unmodified record, independent of export time and display order."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_export(item, evaluator):
    # Never treat UI readiness as authority: reread sources for this request.
    evaluator.require_current(item)
    sources = {}
    if item.get("document_id"):
        doc = evaluator.document(item["document_id"])
        key = (doc["id"], doc["version"], "")
        sources[key] = SourceManifestEntry(
            document_id=doc["id"],
            version=doc["version"],
            title=doc["title"],
            source=doc["category"],
            security_label=doc["security_domain"],
            role="context",
        )
    content = item["analysis"]["report"] if item["kind"] == "review" else item["content"]
    for evidence in content["evidence"]:
        # Legacy official evidence may not carry document IDs. Do not collapse
        # unrelated citations into one manifest entry or invent an identifier.
        key = (
            evidence["document_id"],
            evidence["version"],
            "" if evidence["document_id"] else evidence["evidence_id"],
        )
        if key not in sources:
            sources[key] = SourceManifestEntry(
                document_id=evidence["document_id"],
                version=evidence["version"],
                title=evidence["title"],
                source=evidence["source"],
                security_label=evidence["security_label"],
            )
        entry = sources[key]
        if entry.role == "context":
            entry.role = "context-and-evidence"
        if evidence["evidence_id"] not in entry.evidence_ids:
            entry.evidence_ids.append(evidence["evidence_id"])
    return WorkspaceExport(
        exported_at=now(),
        record=item,
        record_sha256=record_digest(item),
        sources=[sources[key] for key in sorted(sources)],
    )
