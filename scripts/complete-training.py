"""Finish model comparison after the currently running local training process exits."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def status(state, **fields):
    target = ROOT / "artifacts/models/training-status.json"
    temporary = target.with_suffix(".pending.json")
    temporary.write_text(
        json.dumps(
            {
                "state": state,
                "updated_at": datetime.now(UTC).isoformat(),
                **fields,
            },
            indent=2,
        )
    )
    temporary.replace(target)
    print(state, fields, flush=True)


def main():
    os.chdir(ROOT)
    training = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command = process.info["cmdline"] or []
            if "ml.train" in command and "artifacts/models/legal-bert-cuad" in command:
                training.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    if len(training) != 1:
        raise RuntimeError(f"Expected one matching training process, found {len(training)}")
    status("training", pid=training[0].pid)
    training[0].wait()
    model_path = ROOT / "artifacts/models/legal-bert-cuad"
    provenance_path = model_path / "training_provenance.json"
    if not provenance_path.is_file() or not (model_path / "evaluation_metrics.json").is_file():
        raise RuntimeError("Training exited without a complete model artifact")
    weights_path = model_path / "model.safetensors"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        {
            "model_id": f"legal-bert-cuad-{digest(weights_path)[:12]}",
            "trained_at": datetime.now(UTC).isoformat(),
            "base_model_revision": "15b570cbf88259610b082a167dacc190124f60f6",
            "split_sha256": {
                name: digest(ROOT / f"ml/data/comparison/{name}.jsonl")
                for name in ("train", "validation", "test")
            },
            "dataset_sha256": digest(ROOT / "ml/data/CUAD_v1.json"),
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2))
    status("evaluating")
    subprocess.run([sys.executable, "-m", "ml.evaluate_models"], check=True)
    selection = json.loads((ROOT / "artifacts/models/selected.json").read_text())
    status("verifying_selected_model", **selection)
    subprocess.run(
        [sys.executable, "scripts/verify-model-review.py"],
        check=True,
        env={**os.environ, "CLASSIFIER_MODEL_DIR": selection["model_path"]},
    )
    status(
        "complete",
        **selection,
        comparison="artifacts/models/comparison.json",
        live_review="artifacts/model-review-smoke.json",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error_type=type(error).__name__)
        raise
