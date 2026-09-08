"""Enterprise ingestion connectors and deterministic normalization."""

from ingestion.chunking import chunk_text, normalize_document
from ingestion.connectors import Connector, FilesystemConnector
from ingestion.graph import GraphClient, GraphDeltaConnector
from ingestion.models import (
    ChangeEvent,
    ChangeKind,
    ConnectorBatch,
    NormalizedRecord,
    SourceDocument,
)
from ingestion.pipeline import IngestionResult, ingest
from ingestion.store import ApplyResult, InMemoryIngestionStore, Tombstone

__all__ = [
    "ApplyResult",
    "ChangeEvent",
    "ChangeKind",
    "Connector",
    "ConnectorBatch",
    "FilesystemConnector",
    "GraphClient",
    "GraphDeltaConnector",
    "InMemoryIngestionStore",
    "IngestionResult",
    "NormalizedRecord",
    "SourceDocument",
    "Tombstone",
    "chunk_text",
    "ingest",
    "normalize_document",
]
