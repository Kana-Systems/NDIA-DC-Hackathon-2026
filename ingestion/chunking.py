"""Deterministic, content-preserving text chunking."""

from __future__ import annotations

import hashlib
import re

from ingestion.models import NormalizedRecord, SourceDocument, canonical_checksum


def chunk_text(text: str, *, max_chars: int = 1_200) -> tuple[str, ...]:
    """Split text at stable whitespace boundaries with no overlap."""
    if max_chars < 32:
        raise ValueError("max_chars must be at least 32")
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return ()

    chunks: list[str] = []
    remaining = normalized
    while len(remaining) > max_chars:
        boundary = max(
            remaining.rfind("\n\n", 0, max_chars + 1),
            remaining.rfind("\n", 0, max_chars + 1),
            remaining.rfind(" ", 0, max_chars + 1),
        )
        if boundary <= 0:
            boundary = max_chars
        piece = remaining[:boundary].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[boundary:].strip()
    if remaining:
        chunks.append(remaining)
    return tuple(chunks)


def normalize_document(
    document: SourceDocument,
    *,
    max_chars: int = 1_200,
) -> tuple[NormalizedRecord, ...]:
    document = document.normalized()
    records = []
    for index, text in enumerate(chunk_text(document.text, max_chars=max_chars)):
        checksum = canonical_checksum(text)
        identity = f"{document.corpus}\0{document.document_id}\0{index}\0{checksum}"
        record_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        provenance = {**document.provenance, "chunk_index": index}
        records.append(
            NormalizedRecord(
                record_id=record_id,
                corpus=document.corpus,
                document_id=document.document_id,
                version=document.version,
                chunk_index=index,
                text=text,
                security_label=document.security_label,
                acl_principals=document.acl_principals,
                provenance=provenance,
                checksum=checksum,
                document_checksum=document.checksum,
            )
        )
    return tuple(records)
