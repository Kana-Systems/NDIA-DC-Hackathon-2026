"""Create a separate training variant; preserve calibration/selection/test inputs."""

import hashlib
import json
import shutil
from pathlib import Path

from ml.preprocess_cuad import examples, write_jsonl
from ml.train import read_jsonl


def prepare(source=Path("ml/data/comparison"), output=Path("ml/data/improvement")):
    train = read_jsonl(source / "train.jsonl")
    training_ids = {row["document_id"] for row in train}
    raw = Path("ml/data/CUAD_v1.json")
    data = json.loads(raw.read_text())
    data["data"] = [doc for doc in data["data"] if doc["title"] in training_ids]
    config = json.loads((source / "window_config.json").read_text())
    regenerated = examples(data, config["window_chars"], config["stride_chars"], 1)
    assert {r["document_id"] for r in regenerated} == training_ids
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", regenerated)
    for name in (
        "validation.jsonl",
        "test.jsonl",
        "labels.json",
        "window_config.json",
        "cuad_category_domain_mapping.json",
    ):
        shutil.copyfile(source / name, output / name)
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "change": "deterministic spread of negative windows across training contracts",
                "training_documents": len(training_ids),
                "training_windows": len(regenerated),
                "synthetic_federal_cases_used_for_training": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare()
