"""Bounded local experiments; never deploy or overwrite the incumbent model."""

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.tune_models import atomic_json  # noqa: E402


def main():
    os.chdir(ROOT)
    output = ROOT / "artifacts/improvement"
    output.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_XET": "1"}
    jobs = [
        (
            "legal-bert-weighted",
            [
                "-m",
                "ml.train",
                "--model-name",
                "artifacts/models/legal-bert-cuad",
                "--epochs",
                "1",
                "--learning-rate",
                "0.00001",
            ],
        ),
        (
            "roberta-weighted",
            [
                "-m",
                "ml.train",
                "--model-name",
                "FacebookAI/roberta-base",
                "--revision",
                "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
                "--epochs",
                "3",
            ],
        ),
    ]
    try:
        for name, command in jobs:
            model_dir = Path("artifacts/models") / name
            # A completed artifact can be reused after a network/session interruption.
            if (model_dir / "training_provenance.json").is_file():
                continue
            args = [
                sys.executable,
                *command,
                "--training-dir",
                "ml/data/improvement",
                "--model-dir",
                str(model_dir),
                "--checkpoint-dir",
                f"artifacts/checkpoints/{name}",
                "--positive-weight-cap",
                "4",
                "--batch-size",
                "8",
            ]
            atomic_json(
                output / "training-status.json",
                {
                    "state": "training",
                    "candidate": name,
                    "started_at": datetime.now(UTC).isoformat(),
                    "command": args,
                },
            )
            print(f"Training {name}; log: artifacts/improvement/{name}.log", flush=True)
            with (output / f"{name}.log").open("a") as log:
                subprocess.run(
                    args, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=10800
                )
        atomic_json(
            output / "training-status.json",
            {"state": "comparing", "started_at": datetime.now(UTC).isoformat()},
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.tune_models",
                "--models",
                "artifacts/models/legal-bert-cuad",
                "artifacts/models/legal-bert-weighted",
                "artifacts/models/roberta-weighted",
                "artifacts/models/cuad-linear-comparison",
                "--output",
                "artifacts/improvement/final",
                "--report-test",
            ],
            check=True,
            env=env,
            timeout=3600,
        )
        atomic_json(
            output / "training-status.json",
            {
                "state": "complete",
                "promoted": False,
                "completed_at": datetime.now(UTC).isoformat(),
                "comparison": "artifacts/improvement/final/comparison.json",
                "next": "Downstream evaluation and human review required before promotion",
            },
        )
    except Exception as error:
        atomic_json(
            output / "training-status.json",
            {"state": "failed", "error": str(error), "updated_at": datetime.now(UTC).isoformat()},
        )
        raise


if __name__ == "__main__":
    main()
