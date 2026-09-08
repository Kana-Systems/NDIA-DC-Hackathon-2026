#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export GRADIO_ANALYTICS_ENABLED=false
.venv/bin/python -m ruff format --check app evaluation ingestion knowledge ml tests
.venv/bin/python -m ruff check app evaluation ingestion knowledge ml tests
.venv/bin/python -m pytest -q
.venv/bin/python -m evaluation.benchmark
.venv/bin/python -m evaluation.intelligence_benchmark
cd frontend
npm run lint
npm test
npm run build
