"""Gate 1 evaluation on held-out contracts using production-style windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.gate1_classifier import (
    DEFAULT_CRITICAL_LABELS,
    SUPPORTED_POSITIVE_DOCUMENTS,
    SUPPORTED_POSITIVE_WINDOWS,
    evaluate_gate1,
)
from ml.metrics import slice_metrics
from ml.preprocess_cuad import _answer_spans, labels_for_bounds, write_jsonl
from ml.train import read_jsonl
from ml.tune_models import (
    atomic_json,
    digest,
    encode,
    metrics,
    paired_interval,
    probabilities,
)
from ml.windowing import sliding_bounds


def prepare(raw, ids, config):
    """Build annotation-independent windows and label them only after slicing."""

    selected_ids = set(ids)
    rows = []
    for doc in raw["data"]:
        if doc["title"] not in selected_ids:
            continue
        for paragraph_number, paragraph in enumerate(doc["paragraphs"]):
            spans = _answer_spans(paragraph)
            text = paragraph["context"]
            for window_number, (start, end) in enumerate(
                sliding_bounds(
                    len(text),
                    config["window_chars"],
                    config["stride_chars"],
                )
            ):
                labels = labels_for_bounds(spans, start, end)
                rows.append(
                    {
                        "id": (
                            f"{doc['title']}:{paragraph_number}:production-w{window_number:05d}"
                        ),
                        "document_id": doc["title"],
                        "paragraph_number": paragraph_number,
                        "start_char": start,
                        "end_char": end,
                        "text": text[start:end],
                        "labels": labels,
                    }
                )
    return rows


def _candidate_descriptor(path: Path) -> dict:
    descriptor = json.loads(path.read_text())
    if "model_path" not in descriptor and isinstance(descriptor.get("candidate"), dict):
        descriptor = descriptor["candidate"]
    if not descriptor.get("model_path"):
        raise ValueError("candidate artifact must contain model_path")
    return descriptor


def _decision_thresholds(item, model_id, labels):
    threshold_path = item.get("thresholds_path", "")
    if not threshold_path:
        return np.full(len(labels), 0.5), {"kind": "fixed", "value": 0.5}
    tuning = json.loads(Path(threshold_path).read_text())
    if tuning.get("model_id") != model_id:
        raise ValueError("Threshold/model mismatch")
    missing = [label for label in labels if label not in tuning.get("thresholds", {})]
    if missing:
        raise ValueError(f"Threshold artifact is missing labels: {', '.join(missing)}")
    return (
        np.array([tuning["thresholds"][label] for label in labels], dtype=float),
        {
            "kind": "per_label",
            "path": str(threshold_path),
            "sha256": digest(threshold_path),
        },
    )


def _support(expected, records, labels):
    documents = np.asarray([record["document_id"] for record in records])
    return {
        label: {
            "positive_windows": int(expected[:, index].sum()),
            "positive_documents": len(set(documents[expected[:, index]].tolist())),
        }
        for index, label in enumerate(labels)
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", type=Path, default=Path("artifacts/improvement/final/candidate.json")
    )
    parser.add_argument(
        "--incumbent",
        default="artifacts/models/legal-bert-cuad",
        help="Incumbent model directory",
    )
    parser.add_argument("--incumbent-thresholds", type=Path)
    parser.add_argument("--data", type=Path, default=Path("ml/data/comparison"))
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--raw", type=Path, default=Path("ml/data/CUAD_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/improvement/full-contract"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reliability-bins", type=int, default=10)
    parser.add_argument("--critical-label", action="append")
    parser.add_argument(
        "--fail-on-gate",
        "--fail-on-fail",
        dest="fail_on_gate",
        action="store_true",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    heldout_path = args.heldout or args.data / "test.jsonl"
    heldout = read_jsonl(heldout_path)
    if not heldout:
        raise ValueError("held-out evaluation records cannot be empty")
    heldout_ids = {record["document_id"] for record in heldout}
    selected_ids = sorted(heldout_ids)
    config_path = args.data / "window_config.json"
    labels_path = args.data / "labels.json"
    config = json.loads(config_path.read_text())
    labels = json.loads(labels_path.read_text())
    raw = json.loads(args.raw.read_text())
    records = prepare(raw, selected_ids, config)
    evaluated_ids = {record["document_id"] for record in records}
    missing_ids = heldout_ids - evaluated_ids
    if missing_ids:
        raise ValueError(
            "Held-out documents missing from raw source: " + ", ".join(sorted(missing_ids))
        )
    if not records:
        raise ValueError("production windowing produced no evaluation records")
    split_path = args.output / "windows.jsonl"
    write_jsonl(split_path, records)
    candidate = _candidate_descriptor(args.candidate)
    candidates = [
        {
            "model_path": args.incumbent,
            "thresholds_path": str(args.incumbent_thresholds or ""),
        },
        candidate,
    ]
    expected = encode(records, labels)
    documents = [record["document_id"] for record in records]
    results = []
    predictions_by_role = {}
    for role, item in zip(("incumbent", "candidate"), candidates, strict=True):
        scores, model_id, seconds = probabilities(
            item["model_path"], split_path, labels, args.output
        )
        thresholds, threshold_report = _decision_thresholds(item, model_id, labels)
        predicted = scores >= thresholds
        model_metrics = metrics(
            expected,
            predicted,
            labels,
            scores,
            args.reliability_bins,
        )
        result = {
            "role": role,
            "model_path": item["model_path"],
            "model_id": model_id,
            "thresholds": item.get("thresholds_path", ""),
            "threshold_policy": threshold_report,
            "seconds": seconds,
            "inference_seconds": seconds,
            "metrics": model_metrics,
            "slices": slice_metrics(
                expected,
                predicted,
                labels,
                probabilities=scores,
                lengths=records,
                reliability_bin_count=args.reliability_bins,
            ),
        }
        results.append(result)
        predictions_by_role[role] = predicted

    paired_delta = paired_interval(
        expected,
        predictions_by_role["incumbent"],
        predictions_by_role["candidate"],
        documents,
        samples=args.bootstrap_samples,
        seed=args.seed,
        metric="micro_f1",
    )
    metric_keys = (
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_f2",
        "macro_f1",
        "macro_f2",
        "weighted_f1",
        "weighted_f2",
        "exact_match",
        "brier",
        "ece",
    )
    incumbent_metrics = results[0]["metrics"]
    candidate_metrics = results[1]["metrics"]
    metric_deltas = {key: candidate_metrics[key] - incumbent_metrics[key] for key in metric_keys}
    support = _support(expected, records, labels)
    configured_critical = args.critical_label or [
        label for label in DEFAULT_CRITICAL_LABELS if label in labels
    ]
    gate = evaluate_gate1(
        incumbent_metrics,
        candidate_metrics,
        paired_delta,
        critical_labels=configured_critical,
        supported_labels=support,
        minimum_positive_windows=SUPPORTED_POSITIVE_WINDOWS,
        minimum_positive_documents=SUPPORTED_POSITIVE_DOCUMENTS,
    )
    report = {
        "schema_version": "2.0",
        "benchmark": "gate1_classifier_full_contract",
        "dataset": {
            "heldout_path": str(heldout_path),
            "heldout_sha256": digest(heldout_path),
            "provided_heldout_records": len(heldout),
            "document_ids": selected_ids,
            "documents": len(selected_ids),
            "production_windows": len(records),
            "raw_source_path": str(args.raw),
            "raw_source_sha256": digest(args.raw),
            "labels_path": str(labels_path),
            "labels_sha256": digest(labels_path),
            "window_config_path": str(config_path),
            "window_config_sha256": digest(config_path),
            "windowing": {
                "strategy": "fixed_sliding_bounds_independent_of_annotations",
                "window_chars": config["window_chars"],
                "stride_chars": config["stride_chars"],
            },
            "label_rule": "At least half of the shorter of span/window overlaps",
            "per_label_support": support,
        },
        "models": {result["role"]: result for result in results},
        "comparison": {
            "candidate_minus_incumbent": metric_deltas,
            "paired_micro_f1_delta": paired_delta,
            "calibration": {
                "incumbent": {
                    "brier": incumbent_metrics["brier"],
                    "ece": incumbent_metrics["ece"],
                },
                "candidate": {
                    "brier": candidate_metrics["brier"],
                    "ece": candidate_metrics["ece"],
                },
            },
        },
        "gate1": gate,
        "limitation_details": [
            "Previously held-out commercial CUAD contracts; this is a regression benchmark",
            "Window labels are evaluation annotations, not legal or federal ground truth",
            "Gate passage is an engineering signal and never promotes a model automatically",
        ],
        "limitations": (
            "All provided previously held-out commercial contracts; exploratory regression, "
            "not federal/legal accuracy; no automatic model promotion"
        ),
        # Compatibility fields retained for existing report consumers.
        "document_ids": selected_ids,
        "windows": len(records),
        "raw_source_sha256": digest(args.raw),
        "results": results,
        "input_windows": "Full contracts; fixed sliding bounds independent of annotations",
        "label_rule": "At least half of the shorter of span/window overlaps",
    }
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "documents": len(selected_ids),
                "windows": len(records),
                "paired_delta": paired_delta,
                "gate1": {
                    "passed": gate["passed"],
                    "blockers": gate["blockers"],
                },
            },
            indent=2,
        )
    )
    return int(args.fail_on_gate and not gate["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
