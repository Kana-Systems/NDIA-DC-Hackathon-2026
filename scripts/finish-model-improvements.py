"""Wait without LLM polling, then run full-contract and approved live evaluations."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.tune_models import atomic_json  # noqa: E402


def main():
    os.chdir(ROOT)
    status = Path("artifacts/improvement/final-checks-status.json")
    atomic_json(status, {"state": "waiting_for_local_training", "paid_requests_started": False})
    deadline = time.monotonic() + 8 * 3600
    try:
        while time.monotonic() < deadline:
            path = Path("artifacts/improvement/training-status.json")
            training = json.loads(path.read_text()) if path.exists() else {}
            if training.get("state") == "failed":
                raise RuntimeError("Training failed; downstream checks not started")
            if training.get("state") == "complete":
                break
            time.sleep(30)
        else:
            raise TimeoutError("Local training did not finish within eight hours")
        atomic_json(status, {"state": "full_contract_evaluation", "paid_requests_started": False})
        subprocess.run([sys.executable, "-m", "ml.full_contract_eval"], check=True, timeout=3600)
        atomic_json(
            status,
            {
                "state": "live_llm_comparison",
                "paid_requests_started": True,
                "max_review_runs": 36,
                "max_api_attempts": 96,
            },
        )
        subprocess.run(
            [sys.executable, "-m", "evaluation.model_benefit", "--live", "--budget-usd", "10"],
            check=True,
            timeout=14400,
        )
        result = json.loads(Path("artifacts/improvement/llm-benefit/summary.json").read_text())
        atomic_json(
            status,
            {
                "state": "budget_stopped"
                if result.get("budget_stopped")
                else "complete_pending_human_review",
                "automatically_promoted": False,
                "report": "artifacts/improvement/llm-benefit/summary.json",
            },
        )
    except Exception as error:
        atomic_json(
            status, {"state": "failed", "error": str(error), "automatically_promoted": False}
        )
        raise


if __name__ == "__main__":
    main()
