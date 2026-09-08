"""Exploratory full-contract sliding-window regression, with no answer-centered inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.preprocess_cuad import _answer_spans, write_jsonl
from ml.train import read_jsonl
from ml.tune_models import atomic_json, digest, encode, metrics, probabilities
from ml.windowing import sliding_bounds


def prepare(raw, ids, config):
    rows = []
    for doc in raw["data"]:
        if doc["title"] not in ids:
            continue
        for paragraph_number, paragraph in enumerate(doc["paragraphs"]):
            spans = _answer_spans(paragraph)
            text = paragraph["context"]
            for start, end in sliding_bounds(
                len(text), config["window_chars"], config["stride_chars"]
            ):
                labels = {
                    span["label"]
                    for span in spans
                    if max(0, min(end, span["end"]) - max(start, span["start"]))
                    >= 0.5 * min(span["end"] - span["start"], end - start)
                }
                rows.append(
                    {
                        "document_id": doc["title"],
                        "paragraph_number": paragraph_number,
                        "start_char": start,
                        "end_char": end,
                        "text": text[start:end],
                        "labels": sorted(labels),
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", type=Path, default=Path("artifacts/improvement/final/candidate.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/improvement/full-contract"))
    args = parser.parse_args()
    data = Path("ml/data/comparison")
    raw_path = Path("ml/data/CUAD_v1.json")
    test_ids = {r["document_id"] for r in read_jsonl(data / "test.jsonl")}
    selected_ids = sorted(
        test_ids, key=lambda value: hashlib.sha256(("full-v1:" + value).encode()).hexdigest()
    )[:6]
    config = json.loads((data / "window_config.json").read_text())
    labels = json.loads((data / "labels.json").read_text())
    records = prepare(json.loads(raw_path.read_text()), selected_ids, config)
    split_path = args.output / "windows.jsonl"
    write_jsonl(split_path, records)
    candidate = json.loads(args.candidate.read_text())
    candidates = [
        {"model_path": "artifacts/models/legal-bert-cuad", "thresholds_path": ""},
        candidate,
    ]
    results = []
    for item in candidates:
        scores, model_id, seconds = probabilities(
            item["model_path"], split_path, labels, args.output
        )
        thresholds = 0.5
        if item.get("thresholds_path"):
            tuning = json.loads(Path(item["thresholds_path"]).read_text())
            if tuning["model_id"] != model_id:
                raise ValueError("Threshold/model mismatch")
            thresholds = np.array([tuning["thresholds"][label] for label in labels])
        results.append(
            {
                "model_id": model_id,
                "thresholds": item.get("thresholds_path", ""),
                "seconds": seconds,
                "metrics": metrics(encode(records, labels), scores >= thresholds, labels),
            }
        )
    atomic_json(
        args.output / "report.json",
        {
            "document_ids": selected_ids,
            "windows": len(records),
            "raw_source_sha256": digest(raw_path),
            "results": results,
            "input_windows": "Full contracts; fixed sliding bounds independent of annotations",
            "label_rule": "At least half of the shorter of span/window overlaps",
            "limitations": (
                "Six previously held-out commercial contracts; exploratory regression, "
                "not federal/legal accuracy; no model selection from this report"
            ),
        },
    )


if __name__ == "__main__":
    main()
