"""Connector protocol and local/shared-drive filesystem connector."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from ingestion.models import ChangeEvent, ConnectorBatch, SourceDocument, canonical_checksum


@runtime_checkable
class Connector(Protocol):
    """Incremental source connector contract."""

    def changes(self, cursor: str | None = None) -> ConnectorBatch:
        """Return changes after cursor and a durable replacement cursor."""


class FilesystemConnector:
    """Incrementally scan a local path or mounted shared drive."""

    def __init__(
        self,
        root: str | Path,
        *,
        corpus: str,
        security_label: str = "PUBLIC",
        acl_principals: tuple[str, ...] = (),
        metadata: Mapping[str, Mapping[str, object]] | None = None,
        extensions: tuple[str, ...] = (".txt", ".md"),
    ) -> None:
        self.root = Path(root).resolve()
        self.corpus = corpus
        self.security_label = security_label
        self.acl_principals = acl_principals
        self.metadata = metadata or {}
        self.extensions = tuple(extension.casefold() for extension in extensions)

    @staticmethod
    def _decode_cursor(cursor: str | None) -> dict[str, str]:
        if not cursor:
            return {}
        value = json.loads(cursor)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise ValueError("invalid filesystem cursor")
        return value

    def changes(self, cursor: str | None = None) -> ConnectorBatch:
        previous = self._decode_cursor(cursor)
        current: dict[str, str] = {}
        events: list[ChangeEvent] = []
        paths = sorted(
            (
                path
                for path in self.root.rglob("*")
                if path.is_file() and path.suffix.casefold() in self.extensions
            ),
            key=lambda path: path.relative_to(self.root).as_posix(),
        )
        for path in paths:
            relative = path.relative_to(self.root).as_posix()
            text = path.read_text(encoding="utf-8")
            checksum = canonical_checksum(text)
            current[relative] = checksum
            if previous.get(relative) == checksum:
                continue
            item_metadata = self.metadata.get(relative, {})
            label = str(item_metadata.get("security_label", self.security_label))
            principals_value = item_metadata.get("acl_principals", self.acl_principals)
            if isinstance(principals_value, str):
                principals = (principals_value,)
            else:
                principals = tuple(str(value) for value in principals_value)
            events.append(
                ChangeEvent.upsert(
                    SourceDocument(
                        corpus=self.corpus,
                        document_id=relative,
                        version=checksum,
                        text=text,
                        security_label=label,
                        acl_principals=principals,
                        provenance={
                            "connector": "filesystem",
                            "path": relative,
                        },
                    )
                )
            )
        for relative in sorted(previous.keys() - current.keys()):
            events.append(
                ChangeEvent.delete(
                    self.corpus,
                    relative,
                    version=previous[relative],
                    provenance={"connector": "filesystem", "path": relative},
                )
            )
        next_cursor = json.dumps(current, sort_keys=True, separators=(",", ":"))
        return ConnectorBatch(tuple(events), next_cursor)

    def poll(self, cursor: str | None = None) -> ConnectorBatch:
        """Compatibility alias used by schedulers."""
        return self.changes(cursor)
