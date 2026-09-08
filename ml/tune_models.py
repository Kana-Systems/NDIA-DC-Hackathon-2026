"""Validation-only decision tuning and candidate comparison, without deployment."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from ml.inference import model_fn, predict_fn
from ml.train import read_jsonl


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
    calibration = np.array(
        [
            int(hashlib.sha256(("calibration-v1:" + r["document_id"]).encode()).hexdigest(), 16) % 2
            == 0
            for r in records
        ]
    )
    if not calibration.any() or calibration.all():
        raise ValueError("Need distinct calibration and selection documents")
    return calibration, ~calibration


def metrics(y, predicted, labels=None):
    result = {
        "micro_f1": float(f1_score(y, predicted, average="micro", zero_division=0)),
        "micro_f2": float(fbeta_score(y, predicted, beta=2, average="micro", zero_division=0)),
        "precision": float(precision_score(y, predicted, average="micro", zero_division=0)),
        "recall": float(recall_score(y, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "predicted_labels_per_window": float(predicted.sum(axis=1).mean()),
    }
    if labels is not None:
        result["per_label"] = classification_report(
            y, predicted, target_names=labels, output_dict=True, zero_division=0
        )
    return result


def fit_thresholds(y, probabilities, labels, document_ids, precision_floor=0.75):
    """Threshold fitting only: no test labels and no selection labels accepted."""
    grid = np.linspace(0.05, 0.9, 35)
    candidates = [(float(t), metrics(y, probabilities >= t)) for t in grid]
    eligible = [(t, m) for t, m in candidates if m["precision"] >= precision_floor]
    global_t = (
        max(eligible, key=lambda item: (item[1]["micro_f2"], item[0]))[0] if eligible else 0.5
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
                        (fbeta_score(y[:, index], pred, beta=2, zero_division=0), float(t))
                    )
            if options:
                threshold = max(options)[1]
        thresholds[label] = threshold
    return {
        "global_threshold": global_t,
        "thresholds": thresholds,
        "support": support,
        "precision_floor": precision_floor,
        "objective": "calibration F2; rare-label global fallback",
    }


def probabilities(model_path, split_path, labels, output):
    records = read_jsonl(split_path)
    weight_path = Path(model_path) / "model.safetensors"
    if not weight_path.exists():
        weight_path = Path(model_path) / "linear_weights.npz"
    fingerprint = hashlib.sha256(
        (
            digest(weight_path) + digest(split_path) + json.dumps(labels) + "window-inference-v1"
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

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass
    return matrix, model_id, seconds


def paired_interval(y, baseline, candidate, documents, samples=1000):
    """Document-cluster bootstrap, never treat overlapping windows as independent."""
    groups = sorted(set(documents))
    counts = []
    documents = np.array(documents)
    for group in groups:
        mask = documents == group
        pair = []
        for prediction in (baseline, candidate):
            pair.append(
                [
                    np.logical_and(y[mask], prediction[mask]).sum(),
                    np.logical_and(~y[mask], prediction[mask]).sum(),
                    np.logical_and(y[mask], ~prediction[mask]).sum(),
                ]
            )
        counts.append(pair)
    counts = np.array(counts)
    rng = np.random.default_rng(17)
    values = []
    for _ in range(samples):
        totals = counts[rng.integers(0, len(groups), len(groups))].sum(axis=0)
        scores = [5 * tp / max(1, 5 * tp + 4 * fn + fp) for tp, fp, fn in totals]
        values.append(scores[1] - scores[0])
    return {
        "metric": "micro_f2_delta",
        "unit": "document",
        "samples": samples,
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("ml/data/comparison"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/improvement"))
    parser.add_argument("--report-test", action="store_true")
    args = parser.parse_args()
    labels = json.loads((args.data / "labels.json").read_text())
    records = read_jsonl(args.data / "validation.jsonl")
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
            path, args.data / "validation.jsonl", labels, args.output
        )
        tuning = fit_thresholds(y[calibration], scores[calibration], labels, docs[calibration])
        tuning.update(
            {
                "schema_version": "1.0",
                "model_id": model_id,
                "validation_sha256": digest(args.data / "validation.jsonl"),
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
                    "selection": metrics(y[selection], predicted, labels),
                }
            )
    eligible = [i for i, r in enumerate(all_rows) if r["selection"]["precision"] >= 0.75]
    best = max(eligible, key=lambda i: all_rows[i]["selection"]["micro_f2"]) if eligible else 0
    report = {
        "selection_criterion": "selection micro-F2 with precision >= .75; else incumbent",
        "candidates": all_rows,
        "candidate": all_rows[best],
        "paired_selection_delta": paired_interval(
            y[selection], predictions[0], predictions[best], docs[selection]
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
            row["regression_test"] = metrics(encode(test, labels), scores >= thresholds, labels)
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
