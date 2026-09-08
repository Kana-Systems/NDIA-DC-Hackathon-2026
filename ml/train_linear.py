"""Train a real multi-label CUAD classifier with document-disjoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ml.train import read_jsonl


def train(training_dir: Path, output: Path, raw_source: Path) -> dict:
    training = read_jsonl(training_dir / "train.jsonl")
    heldout = read_jsonl(training_dir / "validation.jsonl")
    train_docs = {item["document_id"] for item in training}
    test_docs = {item["document_id"] for item in heldout}
    if train_docs & test_docs:
        raise ValueError("Document leakage between training and held-out evaluation")
    labels = json.loads((training_dir / "labels.json").read_text())
    encoder = MultiLabelBinarizer(classes=labels)
    y_train = encoder.fit_transform([item["labels"] for item in training])
    y_test = encoder.transform([item["labels"] for item in heldout])
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=40_000,
        min_df=2,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(item["text"] for item in training)
    x_test = vectorizer.transform(item["text"] for item in heldout)
    if np.any(y_train.sum(axis=0) == 0):
        raise ValueError("Every label must have training examples")
    classifier = OneVsRestClassifier(
        LogisticRegression(C=4.0, solver="liblinear", max_iter=500, random_state=17),
    )
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    metrics = {
        "micro_f1": f1_score(y_test, predicted, average="micro", zero_division=0),
        "macro_f1": f1_score(y_test, predicted, average="macro", zero_division=0),
        "micro_precision": precision_score(y_test, predicted, average="micro", zero_division=0),
        "micro_recall": recall_score(y_test, predicted, average="micro", zero_division=0),
        "per_label": classification_report(
            y_test,
            predicted,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "threshold": 0.5,
        "all_negative_baseline_micro_f1": 0.0,
    }
    source_hash = hashlib.sha256(raw_source.read_bytes()).hexdigest()
    training_hash = hashlib.sha256((training_dir / "train.jsonl").read_bytes()).hexdigest()
    model_id = f"cuad-tfidf-logreg-{source_hash[:8]}-{training_hash[:8]}"
    provenance = {
        "model_id": model_id,
        "algorithm": "TF-IDF word/bigram features + one-vs-rest logistic regression",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_documents": len(train_docs),
        "heldout_documents": len(test_docs),
        "training_windows": len(training),
        "heldout_windows": len(heldout),
        "training_document_ids": sorted(train_docs),
        "heldout_document_ids": sorted(test_docs),
        "labels": labels,
        "training_data_sha256": training_hash,
        "dataset_sha256": source_hash,
        "source": "https://huggingface.co/datasets/theatticusproject/cuad",
        "attribution": "CUAD, The Atticus Project; Hendrycks et al., NeurIPS 2021",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "modifications": "Labeled positive and negative clause windows; trained classifier",
        "limitations": (
            "Commercial clause classification; not federal compliance or legal confidence"
        ),
        "metrics": metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": model_id,
        "labels": labels,
        "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
        "windowing": json.loads((training_dir / "window_config.json").read_text()),
    }
    np.savez_compressed(
        output / "linear_weights.npz",
        idf=vectorizer.idf_,
        coefficients=np.vstack([estimator.coef_[0] for estimator in classifier.estimators_]),
        intercepts=np.array([estimator.intercept_[0] for estimator in classifier.estimators_]),
    )
    (output / "linear_config.json").write_text(json.dumps(config), encoding="utf-8")
    (output / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(Path(__file__).with_name("CUAD_ATTRIBUTION.md"), output / "CUAD_ATTRIBUTION.md")
    return {
        key: value
        for key, value in provenance.items()
        if key not in {"training_document_ids", "heldout_document_ids", "labels", "metrics"}
    } | {
        "metrics": {key: value for key, value in metrics.items() if key != "per_label"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", type=Path, default=Path("ml/data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/cuad-linear"))
    parser.add_argument("--source", type=Path, default=Path("ml/data/CUAD_v1.json"))
    args = parser.parse_args()
    print(json.dumps(train(args.training_dir, args.output, args.source), indent=2))


if __name__ == "__main__":
    main()
