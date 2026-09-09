import numpy as np
import pytest

from evaluation.calibration import (
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    reliability_bins,
)
from ml.metrics import (
    classification_metrics,
    document_cluster_bootstrap_delta,
    label_frequency_slices,
    length_slices,
)


def test_brier_ece_and_reliability_bin_formulas():
    expected = np.array([[1, 0], [1, 0]], dtype=bool)
    probabilities = np.array([[0.8, 0.2], [0.6, 0.4]])

    assert brier_score(expected, probabilities) == pytest.approx(0.10)
    assert expected_calibration_error(expected, probabilities, n_bins=2) == pytest.approx(0.30)
    bins = reliability_bins(expected, probabilities, n_bins=2)
    assert [item["count"] for item in bins] == [2, 2]
    assert bins[0]["mean_probability"] == pytest.approx(0.30)
    assert bins[0]["observed_frequency"] == 0
    assert bins[1]["mean_probability"] == pytest.approx(0.70)
    assert bins[1]["observed_frequency"] == 1

    report = calibration_metrics(expected, probabilities, n_bins=2, labels=["yes", "no"])
    assert report["brier"] == pytest.approx(0.10)
    assert report["ece"] == pytest.approx(0.30)
    assert report["per_label"]["yes"]["positives"] == 2


def test_multilabel_averages_exact_match_and_per_label_ap():
    expected = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
    predicted = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)
    probabilities = np.array([[0.9, 0.7], [0.8, 0.4], [0.1, 0.8]])

    report = classification_metrics(
        expected,
        predicted,
        ["a", "b"],
        probabilities=probabilities,
        reliability_bin_count=5,
    )
    for average in ("micro", "macro", "weighted"):
        assert report[f"{average}_precision"] == pytest.approx(0.75)
        assert report[f"{average}_recall"] == pytest.approx(0.75)
        assert report[f"{average}_f1"] == pytest.approx(0.75)
        assert report[f"{average}_f2"] == pytest.approx(0.75)
    assert report["exact_match"] == pytest.approx(1 / 3)
    assert report["per_label"]["a"]["average_precision"] == 1
    assert report["per_label"]["b"]["average_precision"] == pytest.approx(5 / 6)


def test_weighted_and_micro_f_scores_use_distinct_confusion_formulas():
    expected = np.array(
        [[1, 0], [1, 0], [1, 0], [0, 1]],
        dtype=bool,
    )
    predicted = np.array(
        [[1, 1], [0, 0], [1, 0], [0, 0]],
        dtype=bool,
    )

    report = classification_metrics(expected, predicted, ["common", "rare"])

    assert report["micro_precision"] == pytest.approx(2 / 3)
    assert report["micro_recall"] == pytest.approx(1 / 2)
    assert report["micro_f1"] == pytest.approx(4 / 7)
    assert report["micro_f2"] == pytest.approx(10 / 19)
    assert report["macro_f1"] == pytest.approx(2 / 5)
    assert report["weighted_f1"] == pytest.approx(3 / 5)
    assert report["macro_f2"] == pytest.approx(5 / 14)
    assert report["weighted_f2"] == pytest.approx(15 / 28)


def test_degenerate_labels_omit_invalid_average_precision():
    expected = np.array(
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    )
    probabilities = np.array(
        [
            [0.1, 0.9, 0.2],
            [0.2, 0.8, 0.7],
            [0.3, 0.7, 0.1],
        ]
    )
    report = classification_metrics(
        expected,
        probabilities >= 0.5,
        ["all_negative", "all_positive", "mixed"],
        probabilities=probabilities,
    )

    assert report["per_label"]["all_negative"]["average_precision"] is None
    assert report["per_label"]["all_negative"]["ap_valid"] is False
    assert report["per_label"]["all_positive"]["average_precision"] is None
    assert report["per_label"]["mixed"]["average_precision"] == 1
    assert report["brier"] >= 0


def test_length_and_label_frequency_slices():
    assert length_slices([100, 700, 3000]) == {
        "short": [0],
        "medium": [1],
        "long": [2],
    }
    expected = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=bool,
    )
    assert label_frequency_slices(
        expected,
        ["rare", "mid", "none", "frequent"],
        rare_max=1,
        frequent_min=5,
    ) == {
        "unsupported": ["none"],
        "rare": ["rare"],
        "mid_frequency": ["mid"],
        "frequent": ["frequent"],
    }


def test_document_cluster_bootstrap_is_deterministic():
    expected = np.array([[1], [0]] * 8, dtype=bool)
    incumbent = np.zeros_like(expected)
    candidate = expected.copy()
    documents = [f"doc-{index // 2}" for index in range(len(expected))]

    first = document_cluster_bootstrap_delta(
        expected,
        incumbent,
        candidate,
        documents,
        samples=100,
        seed=123,
    )
    second = document_cluster_bootstrap_delta(
        expected,
        incumbent,
        candidate,
        documents,
        samples=100,
        seed=123,
    )
    assert first == second
    assert first["documents"] == 8
    assert first["lower_95"] > 0
