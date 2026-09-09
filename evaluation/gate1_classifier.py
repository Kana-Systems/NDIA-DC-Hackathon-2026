"""Fail-closed Gate 1 decision policy for classifier benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

GATE1_THRESHOLDS = {
    "paired_delta_lower_95": 0.0,
    "micro_precision": 0.80,
    "micro_recall": 0.75,
    "macro_f1": 0.50,
}
DEFAULT_CRITICAL_LABELS = (
    "termination_for_convenience",
    "ip_ownership_assignment",
    "uncapped_liability",
)
SUPPORTED_POSITIVE_WINDOWS = 20
SUPPORTED_POSITIVE_DOCUMENTS = 5


def _finite_metric(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, Real) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def _calibration_values(metrics: Mapping[str, Any]) -> tuple[float | None, float | None]:
    nested = metrics.get("calibration")
    calibration = nested if isinstance(nested, Mapping) else metrics
    return (
        _finite_metric(calibration, "brier", "brier_score"),
        _finite_metric(calibration, "ece", "expected_calibration_error"),
    )


def _per_label(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    values = metrics.get("per_label")
    return values if isinstance(values, Mapping) else {}


def _supported_from_metadata(
    label: str,
    metadata: Mapping[str, Any],
    *,
    minimum_positive_windows: int,
    minimum_positive_documents: int,
) -> bool:
    value = metadata.get(label)
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return value >= minimum_positive_windows
    if not isinstance(value, Mapping):
        return False
    if isinstance(value.get("supported"), bool):
        return bool(value["supported"])
    windows = value.get("positive_windows", value.get("support"))
    documents = value.get("positive_documents")
    windows_supported = (
        isinstance(windows, Real) and windows >= minimum_positive_windows
    )
    documents_supported = (
        documents is None
        or (
            isinstance(documents, Real)
            and documents >= minimum_positive_documents
        )
    )
    return windows_supported and documents_supported


def _threshold_checks(
    candidate_metrics: Mapping[str, Any],
    paired_delta: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    checks = {}
    blockers = []
    definitions = (
        (
            "paired_improvement",
            _finite_metric(paired_delta, "lower_95", "lower95", "lower"),
            GATE1_THRESHOLDS["paired_delta_lower_95"],
            True,
            "Paired candidate delta lower 95% bound must be above zero",
        ),
        (
            "micro_precision",
            _finite_metric(candidate_metrics, "micro_precision", "precision"),
            GATE1_THRESHOLDS["micro_precision"],
            False,
            "Candidate micro precision is below 0.80",
        ),
        (
            "micro_recall",
            _finite_metric(candidate_metrics, "micro_recall", "recall"),
            GATE1_THRESHOLDS["micro_recall"],
            False,
            "Candidate micro recall is below 0.75",
        ),
        (
            "macro_f1",
            _finite_metric(candidate_metrics, "macro_f1"),
            GATE1_THRESHOLDS["macro_f1"],
            False,
            "Candidate macro F1 is below 0.50",
        ),
    )
    for name, actual, threshold, strict, message in definitions:
        passed = actual is not None and (actual > threshold if strict else actual >= threshold)
        checks[name] = {
            "passed": passed,
            "actual": actual,
            "threshold": threshold,
            "comparison": ">" if strict else ">=",
        }
        if not passed:
            blockers.append(message)
    return checks, blockers


def _calibration_check(
    incumbent_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    incumbent_brier, incumbent_ece = _calibration_values(incumbent_metrics)
    candidate_brier, candidate_ece = _calibration_values(candidate_metrics)
    values = (incumbent_brier, incumbent_ece, candidate_brier, candidate_ece)
    available = all(value is not None for value in values)
    passed = bool(
        available
        and candidate_brier <= incumbent_brier
        and candidate_ece <= incumbent_ece
        and (candidate_brier < incumbent_brier or candidate_ece < incumbent_ece)
    )
    return {
        "passed": passed,
        "policy": "Brier and ECE non-regression with strict improvement in at least one",
        "incumbent": {"brier": incumbent_brier, "ece": incumbent_ece},
        "candidate": {"brier": candidate_brier, "ece": candidate_ece},
    }


def _supported_critical_labels(
    critical: list[str],
    candidate_labels: Mapping[str, Any],
    supported_labels: Iterable[str] | Mapping[str, Any] | None,
    *,
    minimum_positive_windows: int,
    minimum_positive_documents: int,
) -> set[str]:
    metadata = candidate_labels if supported_labels is None else supported_labels
    if isinstance(metadata, Mapping):
        return {
            label
            for label in critical
            if _supported_from_metadata(
                label,
                metadata,
                minimum_positive_windows=minimum_positive_windows,
                minimum_positive_documents=minimum_positive_documents,
            )
        }
    return set(metadata) & set(critical)


def _critical_label_check(
    incumbent_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    critical: list[str],
    supported: set[str],
    *,
    minimum_positive_windows: int,
    minimum_positive_documents: int,
    critical_metric: str,
    regression_tolerance: float,
) -> tuple[dict[str, Any], list[str]]:
    incumbent_labels = _per_label(incumbent_metrics)
    candidate_labels = _per_label(candidate_metrics)
    metric_keys = (
        (critical_metric, "f1-score")
        if critical_metric in {"f1", "f1-score"}
        else (critical_metric,)
    )
    regressions = []
    comparisons = {}
    for label in critical:
        if label not in supported:
            continue
        incumbent_row = incumbent_labels.get(label)
        candidate_row = candidate_labels.get(label)
        incumbent_value = (
            _finite_metric(incumbent_row, *metric_keys)
            if isinstance(incumbent_row, Mapping)
            else None
        )
        candidate_value = (
            _finite_metric(candidate_row, *metric_keys)
            if isinstance(candidate_row, Mapping)
            else None
        )
        passed = (
            incumbent_value is not None
            and candidate_value is not None
            and candidate_value + regression_tolerance >= incumbent_value
        )
        comparisons[label] = {
            "passed": passed,
            "incumbent": incumbent_value,
            "candidate": candidate_value,
            "delta": (
                candidate_value - incumbent_value
                if incumbent_value is not None and candidate_value is not None
                else None
            ),
        }
        if not passed:
            regressions.append(label)
    return (
        {
            "passed": not regressions,
            "metric": critical_metric,
            "configured_labels": critical,
            "supported_labels": sorted(supported),
            "minimum_positive_windows": minimum_positive_windows,
            "minimum_positive_documents": minimum_positive_documents,
            "comparisons": comparisons,
        },
        regressions,
    )


def evaluate_gate1(
    incumbent_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    paired_delta: Mapping[str, Any],
    *,
    critical_labels: Iterable[str] = DEFAULT_CRITICAL_LABELS,
    supported_labels: Iterable[str] | Mapping[str, Any] | None = None,
    minimum_positive_windows: int = SUPPORTED_POSITIVE_WINDOWS,
    minimum_positive_documents: int = SUPPORTED_POSITIVE_DOCUMENTS,
    critical_metric: str = "recall",
    regression_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Apply the Gate 1 classifier thresholds.

    Calibration is a Pareto check: Brier and ECE must both be no worse than the
    incumbent and at least one must improve strictly. Critical-label safety is
    evaluated on recall by default and only for labels with sufficient support.
    """

    if minimum_positive_windows < 1 or minimum_positive_documents < 1:
        raise ValueError("minimum support thresholds must be positive")
    if regression_tolerance < 0:
        raise ValueError("regression_tolerance cannot be negative")

    checks, blockers = _threshold_checks(candidate_metrics, paired_delta)
    checks["calibration_improves"] = _calibration_check(
        incumbent_metrics,
        candidate_metrics,
    )
    if not checks["calibration_improves"]["passed"]:
        blockers.append("Candidate calibration does not improve over the incumbent")

    critical = list(dict.fromkeys(critical_labels))
    candidate_labels = _per_label(candidate_metrics)
    supported = _supported_critical_labels(
        critical,
        candidate_labels,
        supported_labels,
        minimum_positive_windows=minimum_positive_windows,
        minimum_positive_documents=minimum_positive_documents,
    )
    checks["critical_supported_labels"], regressions = _critical_label_check(
        incumbent_metrics,
        candidate_metrics,
        critical,
        supported,
        minimum_positive_windows=minimum_positive_windows,
        minimum_positive_documents=minimum_positive_documents,
        critical_metric=critical_metric,
        regression_tolerance=regression_tolerance,
    )
    if regressions:
        blockers.append(
            "Candidate regresses supported critical labels: "
            + ", ".join(sorted(regressions))
        )

    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": "1.0",
        "gate": "gate1_classifier",
        "passed": passed,
        "status": "pass" if passed else "fail",
        "recommendation": "pass" if passed else "fail",
        "thresholds": {
            **GATE1_THRESHOLDS,
            "calibration": (
                "Brier and ECE non-regression with strict improvement in at least one"
            ),
            "critical_label_minimum_positive_windows": minimum_positive_windows,
            "critical_label_minimum_positive_documents": minimum_positive_documents,
            "critical_label_regression_tolerance": regression_tolerance,
        },
        "checks": checks,
        "blockers": blockers,
        "automatically_promoted": False,
        "policy": "Engineering benchmark gate; not a legal certification",
    }


gate1_decision = evaluate_gate1
classifier_gate = evaluate_gate1
gate = evaluate_gate1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("incumbent", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("paired_delta", type=Path)
    parser.add_argument("--critical-label", action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-fail", action="store_true")
    args = parser.parse_args()
    report = evaluate_gate1(
        json.loads(args.incumbent.read_text()),
        json.loads(args.candidate.read_text()),
        json.loads(args.paired_delta.read_text()),
        critical_labels=args.critical_label or DEFAULT_CRITICAL_LABELS,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return int(args.fail_on_fail and not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
