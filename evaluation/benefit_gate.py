"""Fail-closed release recommendation; never changes a deployed model registry."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from evaluation.model_benefit import ProvisionalGrade, measure, paired_summary, validate_grade
from ml.tune_models import atomic_json, digest


def gate(summary, *, real_reviewed_cases, distinct_families, artifacts_match, all_reviews_complete):
    blockers = []
    if not artifacts_match:
        blockers.append("Model, corpus, prompts or thresholds changed since evaluation")
    if not all_reviews_complete:
        blockers.append("Missing independent human reviews or failed evaluation runs")
    if real_reviewed_cases < 30 or distinct_families < 10:
        blockers.append("Need at least 30 non-synthetic reviewed cases across 10 contract families")
    interval = summary.get("paired_case_success_delta")
    if not interval or interval["lower_95"] <= 0:
        blockers.append("No positive paired improvement with a lower 95% interval above zero")
    baseline = summary["variants"]["rag"]
    candidate = summary["variants"]["rag_classifier"]
    for metric in (
        "critical_missed",
        "false_findings",
        "unsupported",
        "forbidden_claims",
        "failures",
    ):
        if candidate[metric] > baseline[metric]:
            blockers.append(f"Classifier regresses {metric}")
    if (
        baseline["mean_seconds"]
        and (candidate["mean_seconds"] or 0) > 1.5 * baseline["mean_seconds"]
    ):
        blockers.append("Mean review latency exceeds 1.5x baseline budget")
    baseline_tokens = baseline["review_input_tokens"] + baseline["review_output_tokens"]
    candidate_tokens = candidate["review_input_tokens"] + candidate["review_output_tokens"]
    if not baseline_tokens or candidate_tokens > 1.25 * baseline_tokens:
        blockers.append("Token usage missing or exceeds 1.25x baseline budget")
    return {
        "recommendation": "eligible_for_owner_review" if not blockers else "do_not_promote",
        "blockers": blockers,
        "automatically_promoted": False,
        "policy": "Engineering screening policy, not a legal certification or guarantee",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reviewed-packets", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
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
        if manifest.get("synthetic") is False:
            real_cases.add(row["case_id"])
            families.add(row["family"])
    summary = paired_summary(rows, len({r["case_id"] for r in rows}), manifest["repeats"])
    paths = manifest.get("artifact_paths", {})
    matching = bool(paths) and all(
        Path(path).is_file() and digest(path) == manifest[key] for key, path in paths.items()
    )
    recommendation = gate(
        summary,
        real_reviewed_cases=len(real_cases),
        distinct_families=len(families),
        artifacts_match=matching,
        all_reviews_complete=complete and summary["complete"],
    )
    atomic_json(args.run / "human-benefit-gate.json", recommendation)
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
