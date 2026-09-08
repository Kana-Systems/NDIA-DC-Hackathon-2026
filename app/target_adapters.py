"""Analyst-gated adapters for authoritative target-system integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import yaml

from app.models import ReviewDecision, TargetObject


class TargetSystemAdapter(Protocol):
    def publish(self, target: TargetObject) -> dict[str, Any]: ...


class JsonExportAdapter:
    """Safe demo adapter that performs no external write."""

    def publish(self, target: TargetObject) -> dict[str, Any]:
        if target.status != ReviewDecision.APPROVED:
            raise PermissionError("Only analyst-approved objects may be exported")
        return {
            "adapter": "json-export",
            "external_write_performed": False,
            "object": target.model_dump(mode="json"),
        }


class TargetSchemaRegistry:
    def __init__(
        self,
        path: str = "knowledge/mappings/target_fields.yaml",
    ) -> None:
        self.path = Path(path)

    def allowed_fields(self, object_type: str) -> set[str]:
        if not self.path.is_file() or self.path.stat().st_size > 1_000_000:
            raise ValueError("Target-object schema is unavailable")
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        object_types = payload.get("object_types", {}) if isinstance(payload, dict) else {}
        schema = object_types.get(object_type)
        if not isinstance(schema, dict):
            raise ValueError("Unknown target-object type")
        fields = schema.get("fields", [])
        allowed = {
            str(item.get("name")) for item in fields if isinstance(item, dict) and item.get("name")
        }
        if not allowed:
            raise ValueError("Target-object schema has no fields")
        return allowed
