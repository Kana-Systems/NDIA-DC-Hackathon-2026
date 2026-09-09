#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
  echo "Run bash scripts/setup-local.sh first." >&2
  exit 1
fi
# Explicit local defaults; backend configuration still requires secrets elsewhere.
export WORKSPACE_USERNAME="${WORKSPACE_USERNAME:-judge}"
export WORKSPACE_PASSWORD="${WORKSPACE_PASSWORD:-contract-demo}"
export DEMO_JWT_SECRET="${DEMO_JWT_SECRET:-$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(48))')}"
export BEDROCK_ENABLED="${BEDROCK_ENABLED:-true}"
export MODEL_REVIEW_ENABLED="${MODEL_REVIEW_ENABLED:-true}"
export BEDROCK_TIMEOUT_SECONDS="${BEDROCK_TIMEOUT_SECONDS:-120}"
export LOCAL_CORPUS_PATH="${LOCAL_CORPUS_PATH:-$project_dir/artifacts/knowledge/federal-v2.sqlite}"
if [[ -z "${CLASSIFIER_MODEL_DIR:-}" ]]; then
  export MODEL_SELECTION_PATH="$project_dir/artifacts/models/selected.json"
fi
export CLASSIFIER_MODEL_DIR="${CLASSIFIER_MODEL_DIR:-$project_dir/artifacts/models/cuad-linear}"
export OPENSEARCH_ENDPOINT=""
local_port="${PORT:-8080}"
if [[ ! "$local_port" =~ ^[0-9]{1,5}$ ]] || (( 10#$local_port < 1 || 10#$local_port > 65535 )); then
  echo "PORT must be a number from 1 to 65535." >&2
  exit 1
fi
# Keep local instances on loopback; PORT allows a second checkout without
# stopping another developer's running workspace.
echo "Acquisition Lens: http://127.0.0.1:${local_port}/lens/"
echo "Lens uses the configured WORKSPACE_PASSWORD, or contract-demo by default."
echo "Model review: $MODEL_REVIEW_ENABLED; Bedrock: $BEDROCK_ENABLED. Ctrl+C stops the application."
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$local_port"
