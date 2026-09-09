import copy

import pytest

from evaluation.gate1_classifier import evaluate_gate1


@pytest.fixture
def passing_inputs():
    incumbent = {
        "micro_precision": 0.81,
        "micro_recall": 0.76,
        "macro_f1": 0.51,
        "brier": 0.16,
        "ece": 0.08,
        "per_label": {
            "critical": {"support": 30, "recall": 0.80},
            "rare_critical": {"support": 1, "recall": 1.0},
        },
    }
    candidate = {
        "micro_precision": 0.84,
        "micro_recall": 0.82,
        "macro_f1": 0.55,
        "brier": 0.14,
        "ece": 0.08,
        "per_label": {
            "critical": {"support": 30, "recall": 0.82},
            "rare_critical": {"support": 1, "recall": 0.0},
        },
    }
    interval = {"metric": "micro_f1_delta", "lower_95": 0.01, "upper_95": 0.08}
    return incumbent, candidate, interval


def test_gate1_passes_all_thresholds_and_supported_critical_labels(passing_inputs):
    incumbent, candidate, interval = passing_inputs
    result = evaluate_gate1(
        incumbent,
        candidate,
        interval,
        critical_labels=["critical", "rare_critical"],
        supported_labels={
            "critical": {"positive_windows": 30, "positive_documents": 7},
            "rare_critical": {"positive_windows": 1, "positive_documents": 1},
        },
        minimum_positive_windows=20,
        minimum_positive_documents=5,
    )

    assert result["passed"] is True
    assert result["status"] == "pass"
    assert result["blockers"] == []
    assert result["checks"]["critical_supported_labels"]["supported_labels"] == ["critical"]
    assert result["automatically_promoted"] is False


@pytest.mark.parametrize(
    ("change", "check"),
    [
        (("interval", "lower_95", 0.0), "paired_improvement"),
        (("candidate", "micro_precision", 0.79), "micro_precision"),
        (("candidate", "micro_recall", 0.79), "micro_recall"),
        (("candidate", "macro_f1", 0.49), "macro_f1"),
        (("candidate", "brier", 0.17), "calibration_improves"),
    ],
)
def test_gate1_fails_each_required_threshold(passing_inputs, change, check):
    incumbent, candidate, interval = copy.deepcopy(passing_inputs)
    target, key, value = change
    {"candidate": candidate, "interval": interval}[target][key] = value

    result = evaluate_gate1(incumbent, candidate, interval)

    assert result["passed"] is False
    assert result["checks"][check]["passed"] is False
    assert result["blockers"]


def test_gate1_fails_supported_critical_label_regression(passing_inputs):
    incumbent, candidate, interval = passing_inputs
    candidate["per_label"]["critical"]["recall"] = 0.79

    result = evaluate_gate1(
        incumbent,
        candidate,
        interval,
        critical_labels=["critical"],
    )

    assert result["passed"] is False
    assert result["checks"]["critical_supported_labels"]["passed"] is False
    assert "critical" in result["blockers"][-1]


def test_gate1_fails_closed_when_calibration_is_missing(passing_inputs):
    incumbent, candidate, interval = passing_inputs
    del candidate["ece"]

    result = evaluate_gate1(incumbent, candidate, interval)

    assert result["passed"] is False
    assert result["checks"]["calibration_improves"]["passed"] is False
