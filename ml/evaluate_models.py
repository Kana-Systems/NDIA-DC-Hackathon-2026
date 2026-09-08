"""Select a classifier on validation contracts, then report untouched test results."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from ml.inference import model_fn, predict_fn
from ml.train import read_jsonl


def evaluate(model_path: str, records: list[dict], labels: list[str]) -> dict:
    model = model_fn(model_path)
    predicted = []
    for offset in range(0, len(records), 64):
        batch = records[offset : offset + 64]
        output = predict_fn({"texts": [r["text"] for r in batch], "threshold": 0.5}, model)
        predicted.extend([item["label"] for item in row["labels"]] for row in output["predictions"])
    encoder = MultiLabelBinarizer(classes=labels)
    expected = encoder.fit_transform([row["labels"] for row in records])
    actual = encoder.transform(predicted)
    return {
        "model_path": model_path,
        "model_id": getattr(model, "model_id", "unknown"),
        "micro_f1": f1_score(expected, actual, average="micro", zero_division=0),
        "macro_f1": f1_score(expected, actual, average="macro", zero_division=0),
        "precision": precision_score(expected, actual, average="micro", zero_division=0),
        "recall": recall_score(expected, actual, average="micro", zero_division=0),
        "exact_match": float(np.all(expected == actual, axis=1).mean()),
        "per_label": classification_report(
            expected,
            actual,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "artifacts/models/cuad-linear-comparison",
            "artifacts/models/legal-bert-cuad",
        ],
    )
    args = parser.parse_args()
    root = Path("ml/data/comparison")
    labels = json.loads((root / "labels.json").read_text())
    validation = read_jsonl(root / "validation.jsonl")
    test = read_jsonl(root / "test.jsonl")
    development_ids = {r["document_id"] for r in read_jsonl(root / "train.jsonl") + validation}
    if development_ids & {r["document_id"] for r in test}:
        raise ValueError("Test document leakage")
    scores = [evaluate(path, validation, labels) for path in args.models]
    selected = max(scores, key=lambda score: score["micro_f1"])
    heldout = evaluate(selected["model_path"], test, labels)
    report = {
        "selection_criterion": "Highest validation micro-F1 at fixed threshold 0.5",
        "validation": scores,
        "selected": selected["model_path"],
        "test": heldout,
        "limitations": "Commercial clause windows; not legal or federal compliance accuracy.",
    }
    Path("artifacts/models/comparison.json").write_text(json.dumps(report, indent=2))
    selected_path = Path("artifacts/models/selected.pending.json")
    selected_path.write_text(
        json.dumps(
            {
                "model_path": selected["model_path"],
                "model_id": selected["model_id"],
            },
            indent=2,
        )
    )
    selected_path.replace("artifacts/models/selected.json")
    print(
        json.dumps(
            {
                "selected": selected["model_path"],
                "validation": [
                    {k: v for k, v in row.items() if k != "per_label"} for row in scores
                ],
                "test": {k: v for k, v in heldout.items() if k != "per_label"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
