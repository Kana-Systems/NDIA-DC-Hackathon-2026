import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.model_review import ModelReviewService
from app.models import Evidence
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
from evaluation.benefit_gate import gate
from evaluation.model_benefit import ProvisionalGrade, measure, paired_summary, validate_grade
from ml.federal_data import validate
from ml.full_contract_eval import prepare
from ml.improvement_data import prepare as prepare_improvement
from ml.preprocess_cuad import write_jsonl
from ml.splits import is_calibration_document
from ml.train import positive_weights
from ml.tune_models import fit_thresholds, paired_interval, partition


def test_thresholds_use_supported_labels_and_document_partition():
    records = [{"document_id": f"doc-{i // 2}"} for i in range(40)]
    calibration, selection = partition(records)
    assert not (
        {r["document_id"] for r, flag in zip(records, calibration, strict=True) if flag}
        & {r["document_id"] for r, flag in zip(records, selection, strict=True) if flag}
    )
    y = np.array([[True, False]] * 25 + [[False, True]] * 2 + [[False, False]] * 23)
    scores = np.where(y, 0.35, 0.08)
    result = fit_thresholds(y, scores, ["common", "rare"], [f"d{i}" for i in range(50)])
    assert result["thresholds"]["common"] < 0.5
    assert result["thresholds"]["rare"] == result["global_threshold"]
    assert result["support"]["rare"]["positive_documents"] == 2
    assert result["precision_floor"] == result["recall_floor"] == 0.80
    assert "micro-F1" in result["objective"]


def test_training_checkpoint_validation_reserves_selection_documents(tmp_path):
    source = tmp_path / "comparison"
    output = tmp_path / "improvement"
    source.mkdir()
    validation = [
        {
            "document_id": f"validation-{index}",
            "text": f"validation text {index}",
            "labels": ["payment"] if index % 2 else [],
        }
        for index in range(20)
    ]
    assert any(is_calibration_document(row["document_id"]) for row in validation)
    assert any(not is_calibration_document(row["document_id"]) for row in validation)
    write_jsonl(
        source / "train.jsonl",
        [{"document_id": "training-contract", "text": "Payment terms", "labels": ["payment"]}],
    )
    write_jsonl(source / "validation.jsonl", validation)
    write_jsonl(
        source / "test.jsonl",
        [{"document_id": "test-contract", "text": "Test text", "labels": []}],
    )
    (source / "labels.json").write_text('["payment"]\n')
    (source / "window_config.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "strategy": "answer_centered_positive_sliding_negative",
                "window_chars": 1800,
                "stride_chars": 900,
                "negative_windows_per_positive": 1,
            }
        )
    )
    (source / "cuad_category_domain_mapping.json").write_text("{}\n")
    raw = tmp_path / "CUAD_v1.json"
    raw.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "training-contract",
                        "paragraphs": [
                            {
                                "context": "Payment terms",
                                "qas": [
                                    {
                                        "id": "training-contract__Payment",
                                        "question": "Payment",
                                        "answers": [{"answer_start": 0, "text": "Payment"}],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )
    )

    manifest = prepare_improvement(source, output, raw)
    checkpoint_rows = [
        json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()
    ]

    assert checkpoint_rows
    assert all(is_calibration_document(row["document_id"]) for row in checkpoint_rows)
    assert manifest["checkpoint_validation_documents"] > 0
    assert manifest["selection_validation_documents_reserved"] > 0
    assert manifest["checkpoint_validation_is_document_disjoint_from_selection"] is True


def test_positive_weights_are_bounded_and_no_reweighting_is_identity():
    rows = [{"labels": ["common"]}] * 99 + [{"labels": ["rare"]}]
    assert positive_weights(rows, ["common", "rare"], 4) == [1, 4]
    assert positive_weights(rows, ["common", "rare"], 1) == [1, 1]
    with pytest.raises(ValueError):
        positive_weights(rows, ["common"], 21)


def test_clustered_interval_and_no_synthetic_promotion():
    y = np.array([[True], [False]])
    interval = paired_interval(y, y, y, ["one", "two"], 10)
    assert interval["lower_95"] == interval["upper_95"] == 0
    assert paired_summary([], 6, 2)["promotion_allowed"] is False
    assert paired_summary([], 6, 2)["complete"] is False


def test_benefit_gate_fails_closed_on_synthetic_missing_or_stale_evidence():
    summary = paired_summary([], 6, 2)
    result = gate(
        summary,
        real_reviewed_cases=0,
        distinct_families=0,
        artifacts_match=False,
        all_reviews_complete=False,
    )
    assert result["recommendation"] == "do_not_promote"
    assert result["automatically_promoted"] is False
    assert len(result["blockers"]) >= 4


def test_classifier_off_loads_no_model_and_has_no_hints(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("classifier must not run")

    monkeypatch.setattr("app.model_review.trained_model", forbidden)
    monkeypatch.setattr("app.model_review.predict_fn", forbidden)
    evidence = Evidence(evidence_id="e1", source="FAR", title="Synthetic", excerpt="Source")
    monkeypatch.setattr(
        "app.model_review.FederalCorpusRetrieval",
        lambda path: SimpleNamespace(
            retrieve=lambda queries: [evidence], manifest=lambda: {}, relations=lambda ids: []
        ),
    )
    settings = Settings(bedrock_enabled=True, classifier_enabled=False)
    service = ModelReviewService(settings)
    prompts = []

    def fake(prompt, **kwargs):
        prompts.append(prompt)
        return {"queries": ["termination"], "executive_summary": "Review", "findings": []}

    service.llm = SimpleNamespace(request_json=fake)
    report = service.review(
        DocumentParser(settings).parse("sample.docx", sample_contract_bytes()), sample_metadata()
    )
    assert service.model is None
    assert prompts[0]["classifier_topics"] == []
    assert prompts[1]["classifier_candidates"] == []
    assert not any(c.clause_id.startswith("CUAD:") for c in report.clause_inventory)
    assert len(service.review_trace["calls"]) == 2


def test_llm_only_needs_neither_corpus_nor_classifier(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No retrieval or classification")

    monkeypatch.setattr("app.model_review.trained_model", forbidden)
    monkeypatch.setattr("app.model_review.FederalCorpusRetrieval", forbidden)
    settings = Settings(bedrock_enabled=True)
    service = ModelReviewService(settings, variant="llm_only")
    service.llm = SimpleNamespace(
        request_json=lambda *args, **kwargs: {"executive_summary": "Review", "findings": []}
    )
    report = service.review(
        DocumentParser(settings).parse("sample.docx", sample_contract_bytes()), sample_metadata()
    )
    assert report.evidence == []
    assert len(service.review_trace["calls"]) == 1


def test_threshold_artifact_must_match_model(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"model_id": "wrong", "thresholds": {}}))
    monkeypatch.setattr(
        "app.model_review.trained_model", lambda path: SimpleNamespace(model_id="right")
    )
    with pytest.raises(ValueError, match="different model"):
        ModelReviewService(Settings(bedrock_enabled=True, classifier_thresholds_path=str(path)))


def test_human_label_import_rejects_drafts_and_family_leakage():
    row = {
        "document_id": "a",
        "family_id": "family",
        "text": "Contract excerpt",
        "labels": ["payment"],
        "source_urls": ["https://example.test/contract"],
        "source_version": "v1",
        "reviewed_by": "Reviewer",
        "review_notes": "Reason",
        "reviewed_at": "2026-09-08T12:00:00+00:00",
        "reviewer_type": "human",
        "review_status": "approved",
        "synthetic": False,
        "split": "train",
    }
    assert validate([row], ["payment"]) == [row]
    with pytest.raises(ValueError, match="Draft"):
        validate([{**row, "review_status": "draft"}], ["payment"])
    other = {**row, "document_id": "b", "text": "Different text", "split": "test"}
    with pytest.raises(ValueError, match="leakage"):
        validate([row, other], ["payment"])


def test_full_contract_input_windows_do_not_depend_on_answer_locations():
    raw = {"data": [{"title": "doc", "paragraphs": [{"context": "word " * 100, "qas": []}]}]}
    a = prepare(raw, ["doc"], {"window_chars": 100, "stride_chars": 50})
    other = copy.deepcopy(raw)
    other["data"][0]["paragraphs"][0]["qas"] = [
        {"id": "doc__Payment", "answers": [{"answer_start": 150, "text": "word " * 4}]}
    ]
    b = prepare(other, ["doc"], {"window_chars": 100, "stride_chars": 50})
    assert [row["text"] for row in a] == [row["text"] for row in b]
    assert a[-1]["end_char"] == 500
    assert any(row["labels"] for row in b)


def test_machine_grader_cannot_invent_ids_or_call_empty_review_success():
    case = {"expected_issues": [{"id": "risk", "critical": True}]}
    report = {"findings": []}
    grade = ProvisionalGrade(recommendation_acceptable=True, explanation="No finding")
    assert measure(case, report, grade)["critical_missed"] == 1
    assert measure(case, report, grade)["case_success"] is False
    grade.detected_issue_ids = ["invented"]
    with pytest.raises(ValueError):
        validate_grade(grade, case, report)
