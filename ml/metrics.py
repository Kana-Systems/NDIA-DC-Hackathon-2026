"""Multi-label metrics and deterministic benchmark slicing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.calibration import calibration_metrics


def _binary_matrix(values: object, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix.reshape(0, 0) if matrix.size == 0 else matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional binary array")
    if not (
        np.issubdtype(matrix.dtype, np.bool_) or np.issubdtype(matrix.dtype, np.number)
    ):
        raise TypeError(f"{name} must contain booleans or binary numbers")
    if matrix.size and not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{name} must contain only binary values")
    return matrix.astype(bool, copy=False)


def _probability_matrix(values: object, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1 and shape[1] == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape != shape:
        raise ValueError("probabilities must have the same shape as expected")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must be finite")
    if np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("probabilities must be between zero and one")
    return matrix


def _ratio(numerator: np.ndarray | int, denominator: np.ndarray | int) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=float),
        where=denominator_array != 0,
    )


def _fbeta(
    true_positive: np.ndarray | int,
    false_positive: np.ndarray | int,
    false_negative: np.ndarray | int,
    beta: float,
) -> np.ndarray:
    beta_squared = beta**2
    numerator = (1 + beta_squared) * np.asarray(true_positive)
    denominator = (
        numerator
        + beta_squared * np.asarray(false_negative)
        + np.asarray(false_positive)
    )
    return _ratio(numerator, denominator)


def _average_precision(expected: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = int(expected.sum())
    if positives == 0 or positives == expected.size:
        return None
    order = np.argsort(-probabilities, kind="stable")
    sorted_expected = expected[order].astype(int)
    sorted_probabilities = probabilities[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_probabilities[1:] != sorted_probabilities[:-1]),
        expected.size - 1,
    ]
    cumulative_positive = np.cumsum(sorted_expected)[group_ends]
    precision = cumulative_positive / (group_ends + 1)
    recall = cumulative_positive / positives
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def _validated_labels(labels: Sequence[str] | None, count: int) -> list[str]:
    names = list(labels) if labels is not None else [str(index) for index in range(count)]
    if len(names) != count:
        raise ValueError("labels must match the number of prediction columns")
    if len(set(names)) != len(names):
        raise ValueError("labels must be unique")
    return names


def _macro_average(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _weighted_average(values: np.ndarray, support: np.ndarray) -> float:
    return float(np.average(values, weights=support)) if support.sum() else 0.0


def _per_label_metrics(
    expected: np.ndarray,
    probabilities: np.ndarray | None,
    labels: Sequence[str],
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    support: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    rows = {}
    valid_average_precision = []
    for index, label in enumerate(labels):
        average_precision = (
            _average_precision(expected[:, index], probabilities[:, index])
            if probabilities is not None
            else None
        )
        if average_precision is not None:
            valid_average_precision.append(average_precision)
        rows[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "f1-score": float(f1[index]),
            "f2": float(f2[index]),
            "support": int(support[index]),
            "average_precision": average_precision,
            "ap_valid": average_precision is not None,
        }
    return rows, valid_average_precision


def _attach_calibration(
    result: dict[str, Any],
    expected: np.ndarray,
    probabilities: np.ndarray | None,
    labels: Sequence[str],
    reliability_bin_count: int,
) -> None:
    if probabilities is None or not probabilities.size:
        return
    calibration = calibration_metrics(
        expected,
        probabilities,
        n_bins=reliability_bin_count,
        labels=labels,
    )
    result["brier"] = calibration["brier"]
    result["brier_score"] = calibration["brier"]
    result["ece"] = calibration["ece"]
    result["reliability_bins"] = calibration["reliability_bins"]
    result["calibration"] = calibration
    for label, label_calibration in calibration["per_label"].items():
        result["per_label"][label].update(
            {
                "brier": label_calibration["brier"],
                "ece": label_calibration["ece"],
            }
        )


def classification_metrics(
    expected: object,
    predicted: object,
    labels: Sequence[str] | None = None,
    *,
    probabilities: object | None = None,
    reliability_bin_count: int = 10,
) -> dict[str, Any]:
    """Compute robust binary multi-label metrics.

    Macro averages include every configured label and use zero for undefined
    precision or recall. Weighted averages use positive-label support. Average
    precision is reported only when a label has both positive and negative
    examples, avoiding misleading values and warnings for degenerate labels.
    """

    expected_matrix = _binary_matrix(expected, "expected")
    predicted_matrix = _binary_matrix(predicted, "predicted")
    if expected_matrix.shape != predicted_matrix.shape:
        raise ValueError("expected and predicted must have the same shape")
    label_names = _validated_labels(labels, expected_matrix.shape[1])

    probability_matrix = (
        _probability_matrix(probabilities, expected_matrix.shape)
        if probabilities is not None
        else None
    )
    true_positive = np.logical_and(expected_matrix, predicted_matrix).sum(axis=0)
    false_positive = np.logical_and(~expected_matrix, predicted_matrix).sum(axis=0)
    false_negative = np.logical_and(expected_matrix, ~predicted_matrix).sum(axis=0)
    support = expected_matrix.sum(axis=0)

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _fbeta(true_positive, false_positive, false_negative, beta=1)
    f2 = _fbeta(true_positive, false_positive, false_negative, beta=2)

    micro_true_positive = int(true_positive.sum())
    micro_false_positive = int(false_positive.sum())
    micro_false_negative = int(false_negative.sum())
    micro_precision = float(
        _ratio(micro_true_positive, micro_true_positive + micro_false_positive)
    )
    micro_recall = float(
        _ratio(micro_true_positive, micro_true_positive + micro_false_negative)
    )
    micro_f1 = float(
        _fbeta(
            micro_true_positive,
            micro_false_positive,
            micro_false_negative,
            beta=1,
        )
    )
    micro_f2 = float(
        _fbeta(
            micro_true_positive,
            micro_false_positive,
            micro_false_negative,
            beta=2,
        )
    )
    label_count = expected_matrix.shape[1]
    support_total = int(support.sum())

    exact_match = (
        float(np.all(expected_matrix == predicted_matrix, axis=1).mean())
        if expected_matrix.shape[0]
        else 0.0
    )
    result: dict[str, Any] = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_f2": micro_f2,
        "precision": micro_precision,
        "recall": micro_recall,
        "macro_precision": _macro_average(precision),
        "macro_recall": _macro_average(recall),
        "macro_f1": _macro_average(f1),
        "macro_f2": _macro_average(f2),
        "weighted_precision": _weighted_average(precision, support),
        "weighted_recall": _weighted_average(recall, support),
        "weighted_f1": _weighted_average(f1, support),
        "weighted_f2": _weighted_average(f2, support),
        "exact_match": exact_match,
        "examples": int(expected_matrix.shape[0]),
        "labels": label_count,
        "positive_decisions": support_total,
    }

    per_label, valid_average_precision = _per_label_metrics(
        expected_matrix,
        probability_matrix,
        label_names,
        precision,
        recall,
        f1,
        f2,
        support,
    )
    result["per_label"] = per_label
    result["per_label_average_precision"] = {
        label: values["average_precision"] for label, values in per_label.items()
    }
    result["macro_average_precision"] = (
        float(np.mean(valid_average_precision)) if valid_average_precision else None
    )
    _attach_calibration(
        result,
        expected_matrix,
        probability_matrix,
        label_names,
        reliability_bin_count,
    )
    return result


def multilabel_metrics(
    predicted: Iterable[Iterable[int]], expected: Iterable[Iterable[int]]
) -> dict[str, Any]:
    """Preserve the set-label API while returning the expanded metric report."""

    predicted_items = [list(item) for item in predicted]
    expected_items = [list(item) for item in expected]
    if len(predicted_items) != len(expected_items):
        raise ValueError("predicted and expected collections must have equal length")
    label_order = list(
        dict.fromkeys(label for labels in expected_items for label in labels)
    )
    label_order.extend(
        label
        for labels in predicted_items
        for label in labels
        if label not in label_order
    )
    pairs = [
        (set(predicted), set(expected))
        for predicted, expected in zip(predicted_items, expected_items, strict=True)
    ]
    indices = {label: index for index, label in enumerate(label_order)}
    expected_matrix = np.zeros((len(pairs), len(label_order)), dtype=bool)
    predicted_matrix = np.zeros_like(expected_matrix)
    for row, (predicted_set, expected_set) in enumerate(pairs):
        for label in expected_set:
            expected_matrix[row, indices[label]] = True
        for label in predicted_set:
            predicted_matrix[row, indices[label]] = True
    result = classification_metrics(
        expected_matrix,
        predicted_matrix,
        [str(label) for label in label_order],
    )
    for key in ("micro_precision", "micro_recall", "micro_f1", "exact_match"):
        result[key] = round(result[key], 6)
    result["examples"] = float(len(pairs))
    return result


def length_slices(
    values: Sequence[int | str | Mapping[str, object]],
    *,
    short_max: int = 512,
    medium_max: int = 2048,
    text_key: str = "text",
) -> dict[str, list[int]]:
    """Return row indices for stable short, medium, and long text slices."""

    if short_max < 0 or medium_max <= short_max:
        raise ValueError("length boundaries must satisfy 0 <= short_max < medium_max")
    lengths = []
    for value in values:
        if isinstance(value, Mapping):
            if text_key not in value:
                raise ValueError(f"record is missing {text_key!r}")
            length = len(str(value[text_key]))
        elif isinstance(value, str):
            length = len(value)
        elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            length = int(value)
        else:
            raise TypeError("length slices require lengths, text strings, or records")
        if length < 0:
            raise ValueError("text lengths cannot be negative")
        lengths.append(length)
    return {
        "short": [index for index, length in enumerate(lengths) if length <= short_max],
        "medium": [
            index
            for index, length in enumerate(lengths)
            if short_max < length <= medium_max
        ],
        "long": [index for index, length in enumerate(lengths) if length > medium_max],
    }


def length_slice_masks(
    values: Sequence[int | str | Mapping[str, object]],
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Return boolean masks corresponding to :func:`length_slices`."""

    slices = length_slices(values, **kwargs)
    return {
        name: np.isin(np.arange(len(values)), indices)
        for name, indices in slices.items()
    }


def label_frequency_slices(
    expected: object,
    labels: Sequence[str],
    *,
    rare_max: int = 5,
    frequent_min: int = 20,
) -> dict[str, list[str]]:
    """Group labels by positive-example support."""

    if rare_max < 1 or frequent_min <= rare_max:
        raise ValueError("frequency boundaries must satisfy 1 <= rare_max < frequent_min")
    expected_matrix = _binary_matrix(expected, "expected")
    label_names = list(labels)
    if len(label_names) != expected_matrix.shape[1]:
        raise ValueError("labels must match the number of expected columns")
    support = expected_matrix.sum(axis=0)
    return {
        "unsupported": [
            label for label, count in zip(label_names, support, strict=True) if count == 0
        ],
        "rare": [
            label
            for label, count in zip(label_names, support, strict=True)
            if 0 < count <= rare_max
        ],
        "mid_frequency": [
            label
            for label, count in zip(label_names, support, strict=True)
            if rare_max < count < frequent_min
        ],
        "frequent": [
            label
            for label, count in zip(label_names, support, strict=True)
            if count >= frequent_min
        ],
    }


def _length_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
    labels: list[str],
    lengths: Sequence[int | str | Mapping[str, object]],
    *,
    short_max: int,
    medium_max: int,
    reliability_bin_count: int,
) -> dict[str, Any]:
    report = {}
    for name, indices in length_slices(
        lengths,
        short_max=short_max,
        medium_max=medium_max,
    ).items():
        selected = np.asarray(indices, dtype=int)
        selected_probabilities = (
            probabilities[selected] if probabilities is not None else None
        )
        report[name] = {
            "examples": len(indices),
            "metrics": classification_metrics(
                expected[selected],
                predicted[selected],
                labels,
                probabilities=selected_probabilities,
                reliability_bin_count=reliability_bin_count,
            ),
        }
    return report


def _frequency_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
    labels: list[str],
    *,
    rare_max: int,
    frequent_min: int,
    reliability_bin_count: int,
) -> dict[str, Any]:
    groups = label_frequency_slices(
        expected,
        labels,
        rare_max=rare_max,
        frequent_min=frequent_min,
    )
    indices = {label: index for index, label in enumerate(labels)}
    report = {}
    for name, selected_labels in groups.items():
        columns = [indices[label] for label in selected_labels]
        selected_probabilities = (
            probabilities[:, columns] if probabilities is not None else None
        )
        report[name] = {
            "labels": selected_labels,
            "metrics": classification_metrics(
                expected[:, columns],
                predicted[:, columns],
                selected_labels,
                probabilities=selected_probabilities,
                reliability_bin_count=reliability_bin_count,
            ),
        }
    return report


def slice_metrics(
    expected: object,
    predicted: object,
    labels: Sequence[str],
    *,
    probabilities: object | None = None,
    lengths: Sequence[int | str | Mapping[str, object]] | None = None,
    short_max: int = 512,
    medium_max: int = 2048,
    rare_max: int = 5,
    frequent_min: int = 20,
    reliability_bin_count: int = 10,
) -> dict[str, Any]:
    """Evaluate metrics by text length and by label-support frequency."""

    expected_matrix = _binary_matrix(expected, "expected")
    predicted_matrix = _binary_matrix(predicted, "predicted")
    if expected_matrix.shape != predicted_matrix.shape:
        raise ValueError("expected and predicted must have the same shape")
    probability_matrix = (
        _probability_matrix(probabilities, expected_matrix.shape)
        if probabilities is not None
        else None
    )
    label_names = list(labels)
    if len(label_names) != expected_matrix.shape[1]:
        raise ValueError("labels must match the number of prediction columns")

    if lengths is not None and len(lengths) != expected_matrix.shape[0]:
        raise ValueError("lengths must match the number of examples")
    length_report = (
        _length_metrics(
            expected_matrix,
            predicted_matrix,
            probability_matrix,
            label_names,
            lengths,
            short_max=short_max,
            medium_max=medium_max,
            reliability_bin_count=reliability_bin_count,
        )
        if lengths is not None
        else {}
    )
    frequency_report = _frequency_metrics(
        expected_matrix,
        predicted_matrix,
        probability_matrix,
        label_names,
        rare_max=rare_max,
        frequent_min=frequent_min,
        reliability_bin_count=reliability_bin_count,
    )
    return {"length": length_report, "label_frequency": frequency_report}


def _confusion_metric(
    metric: str,
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    exact_matches: int,
    examples: int,
) -> float:
    if metric == "exact_match":
        return exact_matches / examples if examples else 0.0
    if metric.startswith("micro_"):
        true_positive_value = int(true_positive.sum())
        false_positive_value = int(false_positive.sum())
        false_negative_value = int(false_negative.sum())
        if metric == "micro_precision":
            return float(
                _ratio(
                    true_positive_value,
                    true_positive_value + false_positive_value,
                )
            )
        if metric == "micro_recall":
            return float(
                _ratio(
                    true_positive_value,
                    true_positive_value + false_negative_value,
                )
            )
        beta = 1 if metric == "micro_f1" else 2
        return float(
            _fbeta(
                true_positive_value,
                false_positive_value,
                false_negative_value,
                beta,
            )
        )
    beta = 1 if metric.endswith("_f1") else 2
    values = _fbeta(true_positive, false_positive, false_negative, beta)
    if metric.startswith("macro_"):
        return float(values.mean()) if values.size else 0.0
    support = true_positive + false_negative
    return float(np.average(values, weights=support)) if support.sum() else 0.0


def document_cluster_bootstrap_delta(
    expected: object,
    incumbent: object,
    candidate: object,
    document_ids: Sequence[object],
    *,
    metric: str = "micro_f2",
    samples: int = 1000,
    seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Bootstrap a paired metric delta while keeping document windows clustered."""

    expected_matrix = _binary_matrix(expected, "expected")
    incumbent_matrix = _binary_matrix(incumbent, "incumbent")
    candidate_matrix = _binary_matrix(candidate, "candidate")
    if not (
        expected_matrix.shape == incumbent_matrix.shape == candidate_matrix.shape
    ):
        raise ValueError("expected, incumbent, and candidate must have the same shape")
    if len(document_ids) != expected_matrix.shape[0]:
        raise ValueError("document_ids must match the number of examples")
    if expected_matrix.shape[0] == 0:
        raise ValueError("bootstrap requires at least one example")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    metric_aliases = {
        "f1": "micro_f1",
        "f2": "micro_f2",
        "precision": "micro_precision",
        "recall": "micro_recall",
    }
    metric_name = metric_aliases.get(metric, metric)
    supported_metrics = {
        "micro_f1",
        "micro_f2",
        "micro_precision",
        "micro_recall",
        "macro_f1",
        "macro_f2",
        "weighted_f1",
        "weighted_f2",
        "exact_match",
    }
    if metric_name not in supported_metrics:
        raise ValueError(f"unsupported bootstrap metric: {metric}")

    grouped_indices: dict[object, list[int]] = {}
    for index, document_id in enumerate(document_ids):
        try:
            grouped_indices.setdefault(document_id, []).append(index)
        except TypeError as exc:
            raise TypeError("document IDs must be hashable") from exc
    groups = [
        np.asarray(indices, dtype=int)
        for _, indices in sorted(
            grouped_indices.items(),
            key=lambda item: (type(item[0]).__name__, repr(item[0])),
        )
    ]
    predictions = (incumbent_matrix, candidate_matrix)
    confusion = np.zeros((len(groups), 2, 3, expected_matrix.shape[1]), dtype=np.int64)
    exact_matches = np.zeros((len(groups), 2), dtype=np.int64)
    example_counts = np.zeros(len(groups), dtype=np.int64)
    for group_index, rows in enumerate(groups):
        group_expected = expected_matrix[rows]
        example_counts[group_index] = len(rows)
        for model_index, model_predictions in enumerate(predictions):
            group_predictions = model_predictions[rows]
            confusion[group_index, model_index, 0] = np.logical_and(
                group_expected,
                group_predictions,
            ).sum(axis=0)
            confusion[group_index, model_index, 1] = np.logical_and(
                ~group_expected,
                group_predictions,
            ).sum(axis=0)
            confusion[group_index, model_index, 2] = np.logical_and(
                group_expected,
                ~group_predictions,
            ).sum(axis=0)
            exact_matches[group_index, model_index] = np.all(
                group_expected == group_predictions,
                axis=1,
            ).sum()

    def delta(selected_groups: np.ndarray) -> float:
        totals = confusion[selected_groups].sum(axis=0)
        exact_totals = exact_matches[selected_groups].sum(axis=0)
        examples = int(example_counts[selected_groups].sum())
        scores = [
            _confusion_metric(
                metric_name,
                totals[model_index, 0],
                totals[model_index, 1],
                totals[model_index, 2],
                int(exact_totals[model_index]),
                examples,
            )
            for model_index in range(2)
        ]
        return scores[1] - scores[0]

    point_delta = delta(np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected_groups = rng.integers(0, len(groups), size=len(groups))
        deltas[sample] = delta(selected_groups)
    tail = (1 - confidence) / 2
    lower = float(np.quantile(deltas, tail))
    upper = float(np.quantile(deltas, 1 - tail))
    return {
        "metric": f"{metric_name}_delta",
        "unit": "document",
        "documents": len(groups),
        "samples": samples,
        "seed": seed,
        "confidence": confidence,
        "point_estimate": point_delta,
        "bootstrap_mean": float(deltas.mean()),
        "lower": lower,
        "upper": upper,
        "lower_95": lower,
        "upper_95": upper,
    }


multilabel_classification_metrics = classification_metrics
evaluate_slices = slice_metrics
