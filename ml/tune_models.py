"""Validation-only decision tuning and candidate comparison, without deployment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from ml.inference import model_fn, predict_fn
from ml.metrics import classification_metrics, document_cluster_bootstrap_delta
from ml.splits import is_calibration_document
from ml.train import read_jsonl

VALIDATION_SPLIT = "validation.jsonl"
MINIMUM_PRECISION = 0.80
MINIMUM_RECALL = 0.80


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pending.json")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def encode(records, labels):
    return np.asarray([[label in row["labels"] for label in labels] for row in records], dtype=bool)


def partition(records):
    calibration = np.array([is_calibration_document(r["document_id"]) for r in records])
    if not calibration.any() or calibration.all():
        raise ValueError("Need distinct calibration and selection documents")
    return calibration, ~calibration


def metrics(
    y,
    predicted,
    labels=None,
    probability_scores=None,
    reliability_bin_count=10,
    *,
    probabilities=None,
):
    """Compatibility wrapper around the shared multi-label metric report."""

    if probability_scores is not None and probabilities is not None:
        raise ValueError("provide probability_scores or probabilities, not both")
    score_values = probabilities if probabilities is not None else probability_scores
    result = classification_metrics(
        y,
        predicted,
        labels,
        probabilities=score_values,
        reliability_bin_count=reliability_bin_count,
    )
    result["precision"] = result["micro_precision"]
    result["recall"] = result["micro_recall"]
    predicted_array = np.asarray(predicted)
    if predicted_array.ndim == 1:
        predicted_array = predicted_array.reshape(-1, 1)
    result["predicted_labels_per_window"] = (
        float(predicted_array.sum(axis=1).mean()) if predicted_array.shape[0] else 0.0
    )
    if labels:
        legacy_report = classification_report(
            y,
            predicted,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        result["per_label"].update(
            {key: value for key, value in legacy_report.items() if key not in result["per_label"]}
        )
    return result


def fit_thresholds(
    y,
    probabilities,
    labels,
    document_ids,
    precision_floor=MINIMUM_PRECISION,
    recall_floor=MINIMUM_RECALL,
):
    """Threshold fitting only: no test labels and no selection labels accepted."""
    grid = np.linspace(0.05, 0.9, 35)
    candidates = [(float(t), metrics(y, probabilities >= t)) for t in grid]
    eligible = [
        (threshold, report)
        for threshold, report in candidates
        if report["precision"] >= precision_floor and report["recall"] >= recall_floor
    ]
    precision_eligible = [item for item in candidates if item[1]["precision"] >= precision_floor]
    pool = eligible or precision_eligible
    global_t = (
        max(
            pool,
            key=lambda item: (
                item[1]["micro_f1"],
                item[1]["micro_recall"],
                item[1]["micro_precision"],
                item[0],
            ),
        )[0]
        if pool
        else 0.5
    )
    thresholds, support = {}, {}
    for index, label in enumerate(labels):
        positives = int(y[:, index].sum())
        docs = len({d for d, present in zip(document_ids, y[:, index], strict=True) if present})
        support[label] = {"positive_windows": positives, "positive_documents": docs}
        threshold = global_t
        if positives >= 20 and docs >= 5 and (~y[:, index]).sum() >= 20:
            options = []
            for t in grid:
                pred = probabilities[:, index] >= t
                precision = precision_score(y[:, index], pred, zero_division=0)
                if precision >= precision_floor:
                    options.append(
                        (
                            f1_score(y[:, index], pred, zero_division=0),
                            recall_score(y[:, index], pred, zero_division=0),
                            precision,
                            float(t),
                        )
                    )
            if options:
                threshold = max(options)[3]
        thresholds[label] = threshold
    return {
        "global_threshold": global_t,
        "thresholds": thresholds,
        "support": support,
        "precision_floor": precision_floor,
        "recall_floor": recall_floor,
        "objective": (
            "calibration micro-F1 with precision/recall floors; rare-label global fallback"
        ),
    }


def supported_weight_path(model_path: Path) -> Path:
    model_path = Path(model_path)
    provenance_path = model_path / "training_provenance.json"
    weight_name = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        weight_name = provenance.get("weights_file")
        if weight_name not in (None, "model.safetensors", "adapter_model.safetensors"):
            raise ValueError("model provenance names an unsupported weights file")
    candidates = [
        model_path / weight_name if weight_name else None,
        model_path / "model.safetensors",
        model_path / "adapter_model.safetensors",
        model_path / "linear_weights.npz",
    ]
    weight_path = next((path for path in candidates if path and path.is_file()), None)
    if weight_path is None:
        raise FileNotFoundError(f"no supported model weights found in {model_path}")
    return weight_path


def probabilities(model_path, split_path, labels, output):
    records = read_jsonl(split_path)
    model_path = Path(model_path)
    weight_path = supported_weight_path(model_path)
    fingerprint = hashlib.sha256(
        (
            digest(weight_path)
            + digest(split_path)
            + digest(Path(__file__).with_name("inference.py"))
            + json.dumps(labels)
            + "window-inference-v2"
        ).encode()
    ).hexdigest()
    cache = output / "scores" / f"{fingerprint}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return saved["scores"], str(saved["model_id"]), float(saved["seconds"])
    model = model_fn(str(model_path))
    matrix = np.zeros((len(records), len(labels)), dtype=np.float32)
    indices = {label: i for i, label in enumerate(labels)}
    start = time.monotonic()
    for offset in range(0, len(records), 64):
        batch = records[offset : offset + 64]
        predictions = predict_fn({"texts": [r["text"] for r in batch], "threshold": 0.0}, model)
        for row, prediction in enumerate(predictions["predictions"], offset):
            for item in prediction["labels"]:
                matrix[row, indices[item["label"]]] = item["score"]
        atomic_json(
            output / "progress.json",
            {
                "stage": "scoring",
                "model": str(model_path),
                "split": str(split_path),
                "completed": min(offset + 64, len(records)),
                "total": len(records),
            },
        )
    seconds = time.monotonic() - start
    cache.parent.mkdir(parents=True, exist_ok=True)
    model_id = model.model_id
    np.savez_compressed(cache, scores=matrix, model_id=model_id, seconds=seconds)
    del model
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    return matrix, model_id, seconds


def paired_interval(
    y,
    baseline,
    candidate,
    documents,
    samples=1000,
    *,
    seed=17,
    metric="micro_f1",
):
    """Document-cluster bootstrap, never treat overlapping windows as independent."""
    return document_cluster_bootstrap_delta(
        y,
        baseline,
        candidate,
        documents,
        metric=metric,
        samples=samples,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("ml/data/comparison"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/improvement"))
    parser.add_argument("--report-test", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reliability-bins", type=int, default=10)
    args = parser.parse_args()
    labels = json.loads((args.data / "labels.json").read_text())
    records = read_jsonl(args.data / VALIDATION_SPLIT)
    train = read_jsonl(args.data / "train.jsonl")
    test = read_jsonl(args.data / "test.jsonl")
    groups = [{r["document_id"] for r in part} for part in (train, records, test)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Document leakage")
    calibration, selection = partition(records)
    y = encode(records, labels)
    docs = np.array([r["document_id"] for r in records])
    all_rows, predictions = [], []
    for path in args.models:
        scores, model_id, seconds = probabilities(
            path, args.data / VALIDATION_SPLIT, labels, args.output
        )
        tuning = fit_thresholds(y[calibration], scores[calibration], labels, docs[calibration])
        tuning.update(
            {
                "schema_version": "1.0",
                "model_id": model_id,
                "validation_sha256": digest(args.data / VALIDATION_SPLIT),
                "calibration_document_ids": sorted(set(docs[calibration])),
                "selection_document_ids": sorted(set(docs[selection])),
            }
        )
        tuning_path = args.output / "thresholds" / f"{Path(path).name}.json"
        atomic_json(tuning_path, tuning)
        for kind, thresholds in (
            ("fixed", np.full(len(labels), 0.5)),
            ("tuned", np.array([tuning["thresholds"][label] for label in labels])),
        ):
            predicted = scores[selection] >= thresholds
            predictions.append(predicted)
            all_rows.append(
                {
                    "model_path": path,
                    "model_id": model_id,
                    "kind": kind,
                    "thresholds_path": str(tuning_path) if kind == "tuned" else "",
                    "validation_inference_seconds": seconds,
                    "selection": metrics(
                        y[selection],
                        predicted,
                        labels,
                        scores[selection],
                        args.reliability_bins,
                    ),
                }
            )

    def eligible(index):
        selected = all_rows[index]["selection"]
        return selected["precision"] >= MINIMUM_PRECISION and selected["recall"] >= MINIMUM_RECALL

    incumbent_indices = [
        index for index, row in enumerate(all_rows) if row["model_path"] == args.models[0]
    ]
    eligible_incumbent = [index for index in incumbent_indices if eligible(index)]
    incumbent_best = max(
        eligible_incumbent or incumbent_indices,
        key=lambda index: all_rows[index]["selection"]["micro_f1"],
    )
    eligible_candidates = [index for index in range(len(all_rows)) if eligible(index)]
    best = (
        max(
            eligible_candidates,
            key=lambda index: all_rows[index]["selection"]["micro_f1"],
        )
        if eligible_candidates
        else incumbent_best
    )
    report = {
        "selection_criterion": (
            "selection micro-F1 with precision >= .80 and recall >= .80; else incumbent"
        ),
        "candidates": all_rows,
        "incumbent": all_rows[incumbent_best],
        "candidate": all_rows[best],
        "paired_selection_delta": paired_interval(
            y[selection],
            predictions[incumbent_best],
            predictions[best],
            docs[selection],
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "promoted": False,
        "downstream_llm_benefit": "not established",
        "limitations": [
            "Commercial annotation-centered windows, not whole-contract or legal accuracy",
            "Existing models used validation during checkpoint selection; this is exploratory",
            "Prior test already inspected; repeat test is regression-only, not confirmatory",
        ],
    }
    # Freeze the candidate before touching test outcomes.
    atomic_json(args.output / "candidate.json", all_rows[best])
    if args.report_test:
        for index in dict.fromkeys((0, best)):
            row = all_rows[index]
            scores, _, _ = probabilities(
                row["model_path"], args.data / "test.jsonl", labels, args.output
            )
            thresholds = 0.5
            if row["thresholds_path"]:
                tuning = json.loads(Path(row["thresholds_path"]).read_text())
                thresholds = np.array([tuning["thresholds"][label] for label in labels])
            row["regression_test"] = metrics(
                encode(test, labels),
                scores >= thresholds,
                labels,
                scores,
                args.reliability_bins,
            )
    atomic_json(args.output / "comparison.json", report)
    atomic_json(
        args.output / "progress.json",
        {"stage": "complete", "candidate": all_rows[best]["model_path"], "promoted": False},
    )
    print(
        json.dumps(
            {
                "candidate": all_rows[best]["model_path"],
                "kind": all_rows[best]["kind"],
                "selection": {
                    k: v for k, v in all_rows[best]["selection"].items() if k != "per_label"
                },
                "paired_delta": report["paired_selection_delta"],
                "promoted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
