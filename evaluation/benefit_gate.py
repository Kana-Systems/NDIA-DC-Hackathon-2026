"""Fail-closed release recommendation; never changes a deployed model registry."""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from evaluation.model_benefit import (
    ProvisionalGrade,
    measure,
    paired_summary,
    synthetic_engineering_gate,
    validate_grade,
)
from ml.tune_models import atomic_json, digest


def engineering_gate(summary):
    """Public entry point for the non-production synthetic engineering gate."""

    return synthetic_engineering_gate(summary)


def is_synthetic_evaluation(manifest):
    label_statuses = [
        manifest.get("label_status", ""),
        *manifest.get("benchmark_label_statuses", []),
    ]
    return (
        manifest.get("synthetic") is not False
        or any("synthetic" in str(status).lower() for status in label_statuses)
        or manifest.get("production_evidence") is False
    )


def gate(summary, *, real_reviewed_cases, distinct_families, artifacts_match, all_reviews_complete):
    blockers = []
    if artifacts_match is not True:
        blockers.append("Model, corpus, prompts or thresholds changed since evaluation")
    if all_reviews_complete is not True:
        blockers.append("Missing independent human reviews or failed evaluation runs")
    valid_case_counts = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (real_reviewed_cases, distinct_families)
    )
    if not valid_case_counts or real_reviewed_cases < 30 or distinct_families < 10:
        blockers.append("Need at least 30 non-synthetic reviewed cases across 10 contract families")
    interval = summary.get("paired_case_success_delta")
    lower_95 = interval.get("lower_95") if isinstance(interval, dict) else None
    if (
        not isinstance(lower_95, (int, float))
        or isinstance(lower_95, bool)
        or not math.isfinite(lower_95)
        or lower_95 <= 0
    ):
        blockers.append("No positive paired improvement with a lower 95% interval above zero")
    variants = summary.get("variants", {})
    baseline = variants.get("rag")
    candidate = variants.get("rag_classifier")
    if not baseline or not candidate:
        blockers.append("Missing paired retrieval or classifier evaluation arm")
    else:
        for metric in (
            "critical_missed",
            "false_findings",
            "unsupported",
            "forbidden_claims",
            "failures",
        ):
            baseline_value = baseline.get(metric)
            candidate_value = candidate.get(metric)
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (baseline_value, candidate_value)
            ):
                blockers.append(f"Missing required metric: {metric}")
            elif candidate_value > baseline_value:
                blockers.append(f"Classifier regresses {metric}")
        baseline_seconds = baseline.get("mean_seconds")
        candidate_seconds = candidate.get("mean_seconds")
        valid_seconds = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in (baseline_seconds, candidate_seconds)
        )
        if not valid_seconds:
            blockers.append("Mean review latency is missing or invalid")
        elif baseline_seconds and candidate_seconds > 1.5 * baseline_seconds:
            blockers.append("Mean review latency exceeds 1.5x baseline budget")
        token_values = [
            baseline.get("review_input_tokens"),
            baseline.get("review_output_tokens"),
            candidate.get("review_input_tokens"),
            candidate.get("review_output_tokens"),
        ]
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in token_values
        ):
            blockers.append("Token usage missing or exceeds 1.25x baseline budget")
        else:
            baseline_tokens = token_values[0] + token_values[1]
            candidate_tokens = token_values[2] + token_values[3]
            if not baseline_tokens or candidate_tokens > 1.25 * baseline_tokens:
                blockers.append("Token usage missing or exceeds 1.25x baseline budget")
    return {
        "recommendation": "eligible_for_owner_review" if not blockers else "do_not_promote",
        "blockers": blockers,
        "automatically_promoted": False,
        "production_evidence_required": True,
        "policy": "Engineering screening policy, not a legal certification or guarantee",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reviewed-packets", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    evaluation_is_synthetic = is_synthetic_evaluation(manifest)
    rows, real_cases, families = [], set(), set()
    complete = True
    for path in sorted((args.run / "runs").glob("*.json")):
        row = json.loads(path.read_text())
        packet_path = args.reviewed_packets / path.name
        if not packet_path.exists() or row.get("error") or not row.get("usage_complete"):
            complete = False
            rows.append(row | {"metrics": {}})
            continue
        packet = json.loads(packet_path.read_text())
        if (
            packet.get("independent_human_review") is not True
            or not packet.get("reviewer")
            or not packet.get("reviewed_at")
            or packet.get("run_id") != row["run_id"]
        ):
            complete = False
            rows.append(row | {"metrics": {}})
            continue
        datetime.fromisoformat(packet["reviewed_at"])
        original_packet = json.loads((args.run / "human-review" / path.name).read_text())
        if any(
            packet.get(key) != original_packet.get(key)
            for key in ("case", "findings", "cited_evidence", "reference_passages")
        ):
            raise ValueError("Review packet inputs were altered")
        grade = ProvisionalGrade.model_validate(packet["grade"])
        validate_grade(grade, packet["case"], row["report"])
        row["metrics"] = measure(packet["case"], row["report"], grade)
        rows.append(row)
        if not evaluation_is_synthetic:
            real_cases.add(row["case_id"])
            families.add(row["family"])
    summary = paired_summary(
        rows,
        manifest.get("case_count", len({r["case_id"] for r in rows})),
        manifest["repeats"],
        manifest.get("variants", ("llm_only", "rag", "rag_classifier")),
    )
    paths = manifest.get("artifact_paths", {})
    matching = bool(paths)
    try:
        matching = matching and all(
            key in manifest and Path(path).is_file() and digest(path) == manifest[key]
            for key, path in paths.items()
        )
    except (OSError, TypeError, ValueError):
        matching = False
    recommendation = gate(
        summary,
        real_reviewed_cases=len(real_cases),
        distinct_families=len(families),
        artifacts_match=matching,
        all_reviews_complete=complete and summary["complete"],
    )
    if evaluation_is_synthetic:
        recommendation["synthetic_engineering_gate"] = engineering_gate(summary)
        recommendation["production_evidence"] = False
    atomic_json(args.run / "human-benefit-gate.json", recommendation)
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
