"""Calibration metrics for binary and multi-label classifier scores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_arrays(
    expected: object,
    probabilities: object,
) -> tuple[np.ndarray, np.ndarray]:
    expected_array = np.asarray(expected)
    probability_array = np.asarray(probabilities, dtype=float)
    if expected_array.shape != probability_array.shape:
        raise ValueError("expected and probabilities must have the same shape")
    if expected_array.size == 0:
        raise ValueError("calibration metrics require at least one prediction")
    if not (
        np.issubdtype(expected_array.dtype, np.bool_)
        or np.issubdtype(expected_array.dtype, np.number)
    ):
        raise TypeError("expected values must be booleans or binary numbers")
    if not np.all((expected_array == 0) | (expected_array == 1)):
        raise ValueError("expected values must be binary")
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("probabilities must be finite")
    if np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("probabilities must be between zero and one")
    return expected_array.astype(bool, copy=False), probability_array


def _validate_bin_count(n_bins: int) -> None:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")


def brier_score(expected: object, probabilities: object) -> float:
    """Return the mean squared error of probabilistic binary decisions."""

    expected_array, probability_array = _validated_arrays(expected, probabilities)
    return float(np.mean((probability_array - expected_array.astype(float)) ** 2))


def reliability_bins(
    expected: object,
    probabilities: object,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int | None]]:
    """Return fixed-width reliability bins over all supplied decisions.

    Empty bins are retained so reports from different runs remain directly
    comparable. Intervals are left-closed and right-open, except that the last
    interval includes a probability of one.
    """

    _validate_bin_count(n_bins)
    expected_array, probability_array = _validated_arrays(expected, probabilities)
    outcomes = expected_array.astype(float).ravel()
    scores = probability_array.ravel()
    bin_indices = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    total = scores.size
    bins: list[dict[str, float | int | None]] = []
    for index in range(n_bins):
        mask = bin_indices == index
        count = int(mask.sum())
        mean_probability = float(scores[mask].mean()) if count else None
        observed_frequency = float(outcomes[mask].mean()) if count else None
        gap = (
            abs(mean_probability - observed_frequency)
            if mean_probability is not None and observed_frequency is not None
            else None
        )
        bins.append(
            {
                "index": index,
                "lower_bound": index / n_bins,
                "upper_bound": (index + 1) / n_bins,
                "count": count,
                "weight": count / total,
                "mean_probability": mean_probability,
                "mean_confidence": mean_probability,
                "observed_frequency": observed_frequency,
                "positive_rate": observed_frequency,
                "absolute_gap": gap,
            }
        )
    return bins


def expected_calibration_error(
    expected: object,
    probabilities: object,
    *,
    n_bins: int = 10,
) -> float:
    """Return fixed-width expected calibration error (ECE)."""

    bins = reliability_bins(expected, probabilities, n_bins=n_bins)
    return float(
        sum(
            item["weight"] * item["absolute_gap"]
            for item in bins
            if item["absolute_gap"] is not None
        )
    )


def calibration_metrics(
    expected: object,
    probabilities: object,
    *,
    n_bins: int = 10,
    labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build an aggregate and per-label calibration report."""

    expected_array, probability_array = _validated_arrays(expected, probabilities)
    if expected_array.ndim == 1:
        expected_matrix = expected_array.reshape(-1, 1)
        probability_matrix = probability_array.reshape(-1, 1)
    elif expected_array.ndim == 2:
        expected_matrix = expected_array
        probability_matrix = probability_array
    else:
        raise ValueError("multi-label calibration inputs must be one- or two-dimensional")

    label_names = (
        list(labels)
        if labels is not None
        else [str(index) for index in range(expected_matrix.shape[1])]
    )
    if len(label_names) != expected_matrix.shape[1]:
        raise ValueError("labels must match the number of probability columns")
    if len(set(label_names)) != len(label_names):
        raise ValueError("labels must be unique")

    bins = reliability_bins(expected_matrix, probability_matrix, n_bins=n_bins)
    brier = brier_score(expected_matrix, probability_matrix)
    ece = float(
        sum(
            item["weight"] * item["absolute_gap"]
            for item in bins
            if item["absolute_gap"] is not None
        )
    )
    per_label = {}
    for index, label in enumerate(label_names):
        label_expected = expected_matrix[:, index]
        label_probabilities = probability_matrix[:, index]
        per_label[label] = {
            "positives": int(label_expected.sum()),
            "negatives": int((~label_expected).sum()),
            "brier": brier_score(label_expected, label_probabilities),
            "ece": expected_calibration_error(
                label_expected,
                label_probabilities,
                n_bins=n_bins,
            ),
        }
    return {
        "brier": brier,
        "brier_score": brier,
        "ece": ece,
        "n_bins": n_bins,
        "decisions": int(expected_matrix.size),
        "reliability_bins": bins,
        "per_label": per_label,
    }


calibration_report = calibration_metrics
brier_score_loss = brier_score
ece = expected_calibration_error
