#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  config_path="${RUNPOD_CONFIG_PATH:-$HOME/.runpod/config.toml}"
  if [[ ! -f "$config_path" ]]; then
    echo "Runpod credentials are missing. Run 'flash login' first." >&2
    exit 2
  fi
  RUNPOD_API_KEY="$(
    python3 - "$config_path" <<'PY'
import sys
import tomllib
from pathlib import Path

config = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
key = config.get("apikey") or config.get("default", {}).get("api_key")
if not isinstance(key, str) or not key.strip():
    raise SystemExit("Runpod config contains no API key")
print(key.strip())
PY
  )"
  export RUNPOD_API_KEY
fi

exec "$@"
