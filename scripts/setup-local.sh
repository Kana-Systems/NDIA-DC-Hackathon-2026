#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_for_check="${PYTHON_BIN:-python3}"
if [[ -x .venv/bin/python ]]; then
  python_for_check=.venv/bin/python
fi
"$python_for_check" scripts/verify-shared-artifacts.py
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN:-python3}" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -r ml/requirements-training.txt
cd frontend
npm ci
npm run build
