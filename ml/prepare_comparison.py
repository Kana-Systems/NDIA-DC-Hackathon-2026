"""Reserve document-disjoint validation and test contracts for model selection."""

import hashlib
import shutil
from pathlib import Path

from ml.preprocess_cuad import write_jsonl
from ml.train import read_jsonl


def prepare(source: Path, output: Path) -> None:
    development = read_jsonl(source / "train.jsonl")
    heldout = read_jsonl(source / "validation.jsonl")
    validation_ids = {
        item["document_id"]
        for item in development
        if int(hashlib.sha256(("validation:" + item["document_id"]).encode()).hexdigest(), 16) % 100
        < 20
    }
    training = [item for item in development if item["document_id"] not in validation_ids]
    validation = [item for item in development if item["document_id"] in validation_ids]
    groups = [{item["document_id"] for item in split} for split in (training, validation, heldout)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Split leakage")
    output.mkdir(parents=True, exist_ok=True)
    for name, items in (("train", training), ("validation", validation), ("test", heldout)):
        write_jsonl(output / f"{name}.jsonl", items)
    for filename in ("labels.json", "window_config.json", "cuad_category_domain_mapping.json"):
        shutil.copyfile(source / filename, output / filename)
    print(
        {
            name: {"documents": len(group), "windows": len(items)}
            for name, group, items in zip(
                ("train", "validation", "test"),
                groups,
                (training, validation, heldout),
                strict=True,
            )
        }
    )


if __name__ == "__main__":
    prepare(Path("ml/data/processed"), Path("ml/data/comparison"))
