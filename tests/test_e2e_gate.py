import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.benefit_gate import engineering_gate, gate, is_synthetic_evaluation
from evaluation.model_benefit import (
    ADVERSARIAL_CASE_TYPES,
    MAX_DOCUMENT_CHARACTERS,
    MIN_LONG_DOCUMENT_CHARACTERS,
    BenchmarkDefinition,
    ProvisionalGrade,
    build_schedule,
    load_benchmarks,
    measure,
    metadata,
    paired_summary,
    select_cases,
    validate_execution,
    validate_grade,
)

ROOT = Path(__file__).parents[1]


def metrics(success, *, critical_missed=0, unsupported=0, forbidden_claims=0):
    return {
        "expected": 1,
        "detected": int(success),
        "critical_missed": critical_missed,
        "false_findings": 0,
        "unsupported": unsupported,
        "unsupported_citations": unsupported,
        "cited": 1,
        "supported_citations": int(not unsupported),
        "forbidden_claims": forbidden_claims,
        "recommendation_acceptable": True,
        "case_success": success,
    }


def row(case_id, family, agency, variant, success, **overrides):
    return {
        "case_id": case_id,
        "family": family,
        "agency": agency,
        "repeat": 0,
        "variant": variant,
        "seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "metrics": metrics(success, **overrides),
    }


def passing_summary():
    rows = []
    for case_id, family, agency in (
        ("one", "payment", "Department of Energy"),
        ("two", "cyber", "Department of Defense"),
    ):
        rows.append(row(case_id, family, agency, "rag", False, critical_missed=1))
        rows.append(row(case_id, family, agency, "rag_classifier", True))
    return paired_summary(rows, 2, 1, ("rag", "rag_classifier"))


def test_benchmark_schema_is_synthetic_evaluation_only_and_complete():
    benchmarks, cases = load_benchmarks(
        [
            ROOT / "evaluation" / "federal_cases.json",
            ROOT / "evaluation" / "adversarial_cases.json",
        ]
    )
    adversarial = benchmarks[1]
    assert all(benchmark.training_use == "prohibited" for benchmark in benchmarks)
    assert all(
        benchmark.synthetic and not benchmark.production_evidence for benchmark in benchmarks
    )
    assert all(
        case["split"] == "evaluation"
        and case["training_use"] == "prohibited"
        and case["synthetic"] is True
        and case["production_evidence"] is False
        for case in cases
    )
    assert set(adversarial.required_adversarial_coverage) == set(ADVERSARIAL_CASE_TYPES)
    assert {kind for case in adversarial.cases for kind in case.adversarial_types} == set(
        ADVERSARIAL_CASE_TYPES
    )
    long_case = next(case for case in cases if "long-document" in case["adversarial_types"])
    assert MIN_LONG_DOCUMENT_CHARACTERS <= len(long_case["text"]) <= MAX_DOCUMENT_CHARACTERS
    applicability_case = next(
        case for case in cases if "metadata-applicability" in case["adversarial_types"]
    )
    assert metadata(applicability_case).commercial_product is True
    assert metadata(applicability_case).agency.startswith("National Aeronautics")


def test_schema_rejects_training_or_production_labeling():
    raw = json.loads((ROOT / "evaluation" / "federal_cases.json").read_text())
    with pytest.raises(ValidationError):
        BenchmarkDefinition.model_validate(raw | {"training_use": "allowed"})
    with pytest.raises(ValidationError):
        BenchmarkDefinition.model_validate(raw | {"production_evidence": True})
    with pytest.raises(ValidationError, match="identified as synthetic"):
        BenchmarkDefinition.model_validate(raw | {"label_status": "production_ready"})


def test_synthetic_label_cannot_be_reclassified_by_manifest_flag():
    manifest = {
        "synthetic": False,
        "production_evidence": True,
        "label_status": "synthetic_independent_review_complete",
    }
    assert is_synthetic_evaluation(manifest) is True
    assert (
        is_synthetic_evaluation(
            {
                "synthetic": False,
                "production_evidence": True,
                "label_status": "non_synthetic_human_reviewed",
            }
        )
        is True
    )
    assert (
        is_synthetic_evaluation(
            {
                "synthetic": False,
                "production_evidence": True,
                "label_status": "independent_human_reviewed",
            }
        )
        is False
    )


def test_execution_selection_and_request_bounds():
    _, federal_cases = load_benchmarks([ROOT / "evaluation" / "federal_cases.json"])
    assert validate_execution(federal_cases, 2, ("llm_only", "rag", "rag_classifier")) == 96
    assert len(build_schedule(federal_cases, 2, ("llm_only", "rag", "rag_classifier"))) == 36
    dod = select_cases(federal_cases, agencies=("Department of Defense",))
    assert {case["family"] for case in dod} == {"cyber"}
    assert len(select_cases(federal_cases, families=("payment",), case_limit=1)) == 1
    with pytest.raises(ValueError, match="Unknown case IDs"):
        select_cases(federal_cases, case_ids=("not-a-case",))
    with pytest.raises(ValueError, match="above the configured"):
        validate_execution(federal_cases, 3, ("llm_only", "rag", "rag_classifier"))
    with pytest.raises(ValueError, match="unique"):
        validate_execution(federal_cases, 1, ("rag", "rag"))


def test_family_and_agency_slices_and_synthetic_gate():
    summary = passing_summary()
    assert set(summary["slices"]["family"]) == {"payment", "cyber"}
    assert set(summary["slices"]["agency"]) == {
        "Department of Defense",
        "Department of Energy",
    }
    result = engineering_gate(summary)
    assert result["status"] == "pass"
    assert result["production_evidence"] is False
    assert result["production_eligible"] is False
    assert result["automatically_promoted"] is False


def test_citation_grading_is_exhaustive_and_fail_closed():
    case = {"expected_issues": []}
    report = {"findings": [{"finding_id": "f-1", "citation_ids": ["e-1"]}]}
    omitted = ProvisionalGrade(
        recommendation_acceptable=True,
        explanation="Citation not classified",
    )
    with pytest.raises(ValueError, match="omitted"):
        validate_grade(omitted, case, report)
    assert measure(case, report, omitted)["unsupported_citations"] == 1
    supported = omitted.model_copy(update={"citation_supported_finding_ids": ["f-1"]})
    validate_grade(supported, case, report)
    assert measure(case, report, supported)["unsupported_citations"] == 0


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ({"critical_missed": 1}, "critical misses"),
        ({"unsupported_citations": 1, "unsupported": 1}, "unsupported citations"),
        ({"forbidden_claims": 1}, "forbidden claims"),
    ],
)
def test_synthetic_gate_rejects_candidate_safety_failures(change, blocker):
    summary = passing_summary()
    summary["variants"]["rag_classifier"].update(change)
    result = engineering_gate(summary)
    assert result["status"] == "fail"
    assert any(blocker in item for item in result["blockers"])
    assert result["production_eligible"] is False


def test_synthetic_gate_rejects_failures_nonpositive_benefit_and_incomplete_runs():
    summary = passing_summary()
    summary["variants"]["rag"]["failures"] = 1
    summary["paired_case_success_delta"]["lower_95"] = 0
    summary["complete"] = False
    result = engineering_gate(summary)
    assert result["status"] == "fail"
    assert len(result["blockers"]) == 3


def test_human_release_gate_remains_fail_closed_for_synthetic_evidence():
    summary = passing_summary()
    result = gate(
        summary,
        real_reviewed_cases=0,
        distinct_families=0,
        artifacts_match=True,
        all_reviews_complete=True,
    )
    assert result["recommendation"] == "do_not_promote"
    assert result["automatically_promoted"] is False
    assert result["production_evidence_required"] is True
    reviewed = gate(
        summary,
        real_reviewed_cases=30,
        distinct_families=10,
        artifacts_match=True,
        all_reviews_complete=True,
    )
    assert reviewed["recommendation"] == "eligible_for_owner_review"
    summary["paired_case_success_delta"]["lower_95"] = float("nan")
    malformed = gate(
        summary,
        real_reviewed_cases=30,
        distinct_families=10,
        artifacts_match=True,
        all_reviews_complete=True,
    )
    assert malformed["recommendation"] == "do_not_promote"
