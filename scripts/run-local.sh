#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
  echo "Run bash scripts/setup-local.sh first." >&2
  exit 1
fi
# Explicit local defaults; backend configuration still requires secrets elsewhere.
export GRADIO_USERNAME="${GRADIO_USERNAME:-judge}"
export GRADIO_PASSWORD="${GRADIO_PASSWORD:-contract-demo}"
export DEMO_JWT_SECRET="${DEMO_JWT_SECRET:-$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))')}"
export BEDROCK_ENABLED="${BEDROCK_ENABLED:-true}"
export MODEL_REVIEW_ENABLED="${MODEL_REVIEW_ENABLED:-true}"
export BEDROCK_TIMEOUT_SECONDS="${BEDROCK_TIMEOUT_SECONDS:-120}"
export LOCAL_CORPUS_PATH="${LOCAL_CORPUS_PATH:-$project_dir/artifacts/knowledge/federal-v2.sqlite}"
if [[ -z "${CLASSIFIER_MODEL_DIR:-}" ]]; then
  export MODEL_SELECTION_PATH="$project_dir/artifacts/models/selected.json"
fi
export CLASSIFIER_MODEL_DIR="${CLASSIFIER_MODEL_DIR:-$project_dir/artifacts/models/cuad-linear}"
export GRADIO_ANALYTICS_ENABLED=false
export OPENSEARCH_ENDPOINT=""
echo "Acquisition Lens: http://127.0.0.1:8080/lens/"
echo "PDF/DOCX and advanced workflows: http://127.0.0.1:8080/ui/"
echo "Username: $GRADIO_USERNAME (Gradio only). Use the configured GRADIO_PASSWORD, or contract-demo by default."
echo "Model review: $MODEL_REVIEW_ENABLED; Bedrock: $BEDROCK_ENABLED. Ctrl+C stops the application."
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
