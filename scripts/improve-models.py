"""Run the bounded one-GPU matrix; never deploy or overwrite the incumbent."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.runpod_matrix import main as matrix_main  # noqa: E402


def main() -> int:
    return matrix_main()


if __name__ == "__main__":
    raise SystemExit(main())
