"""Stable document-level partitions shared by training and model selection."""

from __future__ import annotations

import hashlib

CALIBRATION_PARTITION_NAMESPACE = "calibration-v1:"


def is_calibration_document(document_id: str) -> bool:
    """Assign every window from a document to the same stable tuning partition."""

    if not isinstance(document_id, str) or not document_id:
        raise ValueError("document_id must be a non-empty string")
    digest = hashlib.sha256(
        (CALIBRATION_PARTITION_NAMESPACE + document_id).encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % 2 == 0
