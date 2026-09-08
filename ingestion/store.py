"""Idempotent in-memory manifest and record store for ingestion tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ingestion.chunking import normalize_document
from ingestion.models import ChangeEvent, ChangeKind, NormalizedRecord


@dataclass(frozen=True)
class Tombstone:
    corpus: str
    document_id: str
    version: str
    provenance: dict[str, object]


@dataclass(frozen=True)
class ApplyResult:
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


class InMemoryIngestionStore:
    """Atomic-enough reference store with document-level idempotency."""

    def __init__(self, *, max_chunk_chars: int = 1_200) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.records: dict[str, NormalizedRecord] = {}
        self.manifest: dict[tuple[str, str], str] = {}
        self.document_records: dict[tuple[str, str], tuple[str, ...]] = {}
        self.tombstones: dict[tuple[str, str], Tombstone] = {}

    @staticmethod
    def _fingerprint(event: ChangeEvent) -> str:
        document = event.document
        if document is None:
            raise ValueError("upsert event must include a document")
        payload = {
            "version": document.version,
            "security_label": document.security_label,
            "acl_principals": document.acl_principals,
            "provenance": document.provenance,
            "checksum": document.checksum,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def apply(self, events: tuple[ChangeEvent, ...] | list[ChangeEvent]) -> ApplyResult:
        inserted = updated = deleted = unchanged = 0
        for event in events:
            key = event.key
            if event.kind == ChangeKind.DELETE:
                existed = key in self.manifest
                for record_id in self.document_records.pop(key, ()):
                    self.records.pop(record_id, None)
                self.manifest.pop(key, None)
                self.tombstones[key] = Tombstone(
                    corpus=key[0],
                    document_id=key[1],
                    version=event.version,
                    provenance=dict(event.provenance),
                )
                if existed:
                    deleted += 1
                else:
                    unchanged += 1
                continue

            if event.document is None:
                raise ValueError("upsert event must include a document")
            document = event.document.normalized()
            event = ChangeEvent.upsert(document)
            fingerprint = self._fingerprint(event)
            if self.manifest.get(key) == fingerprint:
                unchanged += 1
                continue
            existed = key in self.manifest
            new_records = normalize_document(document, max_chars=self.max_chunk_chars)
            for record_id in self.document_records.get(key, ()):
                self.records.pop(record_id, None)
            for record in new_records:
                self.records[record.record_id] = record
            self.document_records[key] = tuple(record.record_id for record in new_records)
            self.manifest[key] = fingerprint
            self.tombstones.pop(key, None)
            if existed:
                updated += 1
            else:
                inserted += 1
        return ApplyResult(inserted, updated, deleted, unchanged)

    def records_for(self, corpus: str, document_id: str) -> tuple[NormalizedRecord, ...]:
        ids = self.document_records.get((corpus, document_id), ())
        return tuple(self.records[record_id] for record_id in ids)
