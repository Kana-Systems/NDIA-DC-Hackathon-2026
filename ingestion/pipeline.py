"""Connector-to-store orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ingestion.connectors import Connector
from ingestion.models import ChangeEvent
from ingestion.store import ApplyResult


class IngestionStore(Protocol):
    def apply(
        self,
        events: tuple[ChangeEvent, ...] | list[ChangeEvent],
    ) -> ApplyResult: ...


@dataclass(frozen=True)
class IngestionResult:
    cursor: str
    changes: int
    applied: ApplyResult


def ingest(
    connector: Connector,
    store: IngestionStore,
    *,
    cursor: str | None = None,
) -> IngestionResult:
    batch = connector.changes(cursor)
    return IngestionResult(batch.cursor, len(batch.events), store.apply(batch.events))
