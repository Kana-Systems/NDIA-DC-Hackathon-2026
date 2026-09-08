"""Deterministic offline fallback for the prototype classifier."""

from __future__ import annotations

import re
from typing import Any

LABEL_PATTERNS = {
    "cybersecurity": (
        r"\b(?:covered defense information|CUI|cyber incident|NIST SP 800-171)\b",
        r"\b252\.204-7012\b",
    ),
    "sam_registration": (r"\bSystem for Award Management\b", r"\b52\.204-7\b"),
    "termination": (r"\bterminat(?:e|ion)\b",),
    "data_rights": (r"\btechnical data\b", r"\bdata rights\b"),
}


class HeuristicClassifier:
    """Keyword/regex baseline implementing the same output shape as inference."""

    model_id = "local-heuristic-v1"

    def predict(self, text: str, threshold: float = 0.5) -> dict[str, Any]:
        labels = []
        for label, patterns in LABEL_PATTERNS.items():
            matches = sorted(
                {
                    match.group(0)
                    for pattern in patterns
                    for match in re.finditer(pattern, text, re.I)
                }
            )
            score = min(0.99, 0.55 + 0.1 * len(matches)) if matches else 0.05
            if score >= threshold:
                labels.append({"label": label, "score": round(score, 4), "evidence": matches})
        return {"model_id": self.model_id, "labels": labels}

    def predict_many(self, texts: list[str], threshold: float = 0.5) -> list[dict[str, Any]]:
        return [self.predict(text, threshold) for text in texts]
