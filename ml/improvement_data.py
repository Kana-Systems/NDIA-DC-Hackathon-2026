"""Create a separate training variant; preserve calibration/selection/test inputs."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ml.preprocess_cuad import examples, write_jsonl
from ml.train import read_jsonl


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(
    source: Path = Path("ml/data/comparison"),
    output: Path = Path("ml/data/improvement"),
    raw: Path = Path("ml/data/CUAD_v1.json"),
) -> dict[str, Any]:
    """Regenerate the training-only variant and return its immutable manifest."""
    source = Path(source)
    output = Path(output)
    raw = Path(raw)
    required = (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "labels.json",
        "window_config.json",
        "cuad_category_domain_mapping.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"comparison data is incomplete: {', '.join(missing)}")
    if not raw.is_file():
        raise FileNotFoundError(f"raw CUAD data not found: {raw}")

    train = read_jsonl(source / "train.jsonl")
    training_ids = {row["document_id"] for row in train}
    if not training_ids:
        raise ValueError("comparison training data must contain documents")
    data = json.loads(raw.read_text(encoding="utf-8"))
    data["data"] = [doc for doc in data["data"] if doc["title"] in training_ids]
    config = json.loads((source / "window_config.json").read_text(encoding="utf-8"))
    regenerated = examples(data, config["window_chars"], config["stride_chars"], 1)
    regenerated_ids = {row["document_id"] for row in regenerated}
    if regenerated_ids != training_ids:
        missing_ids = sorted(training_ids - regenerated_ids)
        extra_ids = sorted(regenerated_ids - training_ids)
        raise ValueError(
            f"regenerated training documents differ; missing={missing_ids}, extra={extra_ids}"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", regenerated)
    for name in required[1:]:
        shutil.copyfile(source / name, output / name)
    manifest = {
        "schema_version": "2.0",
        "raw_cuad_path": str(raw),
        "raw_cuad_sha256": sha256(raw),
        # Retain the original field for callers that consumed the v1 manifest.
        "source_sha256": sha256(raw),
        "source_files_sha256": {name: sha256(source / name) for name in required},
        "output_files_sha256": {name: sha256(output / name) for name in required},
        "preparation_code_sha256": sha256(Path(__file__)),
        "change": "deterministic spread of negative windows across training contracts",
        "training_documents": len(training_ids),
        "training_windows": len(regenerated),
        "validation_and_test_copied_without_modification": True,
        "synthetic_federal_cases_used_for_training": False,
    }
    target = output / "preparation.json"
    temporary = target.with_suffix(".json.pending")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return manifest


if __name__ == "__main__":
    prepare()
