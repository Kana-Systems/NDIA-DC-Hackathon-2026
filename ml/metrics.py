"""Dependency-free multi-label metrics used by training and benchmarks."""

from __future__ import annotations

from collections.abc import Iterable


def multilabel_metrics(
    predicted: Iterable[Iterable[int]], expected: Iterable[Iterable[int]]
) -> dict[str, float]:
    predicted_items = list(predicted)
    expected_items = list(expected)
    if len(predicted_items) != len(expected_items):
        raise ValueError("predicted and expected collections must have equal length")
    pairs = [
        (set(predicted), set(expected))
        for predicted, expected in zip(predicted_items, expected_items, strict=True)
    ]
    true_positive = sum(len(p & e) for p, e in pairs)
    false_positive = sum(len(p - e) for p, e in pairs)
    false_negative = sum(len(e - p) for p, e in pairs)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(p == e for p, e in pairs) / len(pairs) if pairs else 0.0
    return {
        "micro_precision": round(precision, 6),
        "micro_recall": round(recall, 6),
        "micro_f1": round(f1, 6),
        "exact_match": round(exact, 6),
        "examples": float(len(pairs)),
    }
