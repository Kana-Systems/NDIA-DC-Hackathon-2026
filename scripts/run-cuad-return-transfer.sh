#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <maud|contract-4k|contract-8k>" >&2
  exit 64
fi

mode="$1"
project="${PROJECT_ROOT:-/workspace/project-external-20260909}"
venv="${VENV_ROOT:-/workspace/venv}"
python="${venv}/bin/python"
source_root="/workspace/external-transfer-20260909"
run_root="/workspace/cuad-return-20260909"
training_dir="${CUAD_DATA_ROOT:-/workspace/data/ndia-cuad-return-data}"
base_revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
seed=17

case "$mode" in
  maud)
    source_run="maud-llama-r128-seed17-full"
    candidate="cuad-after-maud-r128-lr050"
    source_description="CUAD v1 after MAUD v1 LoRA-backbone transfer"
    ;;
  contract-4k)
    source_run="contract-llama-4k-r128-seed17-full"
    candidate="cuad-after-contractnli-4k-r128-lr050"
    source_description="CUAD v1 after ContractNLI 4K LoRA-backbone transfer"
    ;;
  contract-8k)
    source_run="contract-llama-8k-r128-seed29-full"
    candidate="cuad-after-contractnli-8k-r128-lr050"
    source_description="CUAD v1 after ContractNLI 8K LoRA-backbone transfer"
    ;;
  *)
    echo "unknown return-transfer mode: $mode" >&2
    exit 64
    ;;
esac

if [[ ! -x "$python" ]]; then
  echo "training Python is missing: $python" >&2
  exit 1
fi

heartbeat="${source_root}/status/${source_run}.json"
source_model="${source_root}/models/${source_run}"
run_id="${candidate}-seed${seed}-full"
model_dir="${run_root}/models/${run_id}"
checkpoint_dir="${run_root}/checkpoints/${run_id}"
status_file="${run_root}/status/${run_id}.json"
telemetry_file="${run_root}/status/${run_id}.jsonl"
log_file="${run_root}/logs/${run_id}.log"
summary_file="${checkpoint_dir}/run-summary.json"

mkdir -p "${run_root}/models" "${run_root}/checkpoints" "${run_root}/status" "${run_root}/logs"
if [[ -f "$summary_file" ]] &&
  [[ "$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$summary_file")" == "complete" ]]; then
  echo "return-transfer run already complete: $run_id"
  exit 0
fi
if [[ -e "$model_dir" || -e "$checkpoint_dir" ]]; then
  echo "refusing to replace partial return-transfer run: $run_id" >&2
  exit 1
fi

deadline=$((SECONDS + 10800))
while true; do
  if [[ -f "$heartbeat" ]]; then
    state="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "$heartbeat")"
    if [[ "$state" == "complete" ]]; then
      break
    fi
    if (( $(date +%s) - $(stat -c %Y "$heartbeat") > 900 )); then
      echo "source heartbeat is stale: $source_run" >&2
      exit 1
    fi
    echo "waiting for source run $source_run (state=$state)"
  else
    echo "waiting for source heartbeat: $source_run"
  fi
  if (( SECONDS >= deadline )); then
    echo "timed out waiting for source run: $source_run" >&2
    exit 1
  fi
  sleep 30
done

required_adapter_files=(
  adapter_config.json
  adapter_model.safetensors
  label_mapping.json
  training_provenance.json
)
for required in "${required_adapter_files[@]}"; do
  if [[ ! -f "${source_model}/${required}" ]]; then
    echo "source adapter is incomplete: ${source_model}/${required}" >&2
    exit 1
  fi
done

expected_train="4e8931f1ee68f11590c6442f379036ea3b4cf3a7e82f33a5b30cbe94b2e07577"
expected_validation="53b78062c7840c9e237ea344d134a76019b6909235ca5dd0d7a6f97cc8ff8f27"
expected_test="99a6c9bafe80151c2d459e1a5f5408a0b13093fa78f442d56b6eefb036a0624a"
printf '%s  %s\n' \
  "$expected_train" "${training_dir}/train.jsonl" \
  "$expected_validation" "${training_dir}/validation.jsonl" \
  "$expected_test" "${training_dir}/test.jsonl" |
  sha256sum --check --strict

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/hf-hub-cache-llama31}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED="$seed"
export PYTHONPATH="$project"

{
  echo "starting CUAD return-transfer run: $run_id"
  "$python" -m ml.train \
    --model-name meta-llama/Llama-3.1-8B \
    --revision "$base_revision" \
    --training-dir "$training_dir" \
    --model-dir "$model_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    --problem-type multi_label \
    --epochs 3 \
    --batch-size 32 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-5 \
    --max-length 512 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --positive-weight-cap 3.0 \
    --metric-for-best-model micro_f1_tuned \
    --evaluation-strategy epoch \
    --early-stopping-patience 1 \
    --precision bf16 \
    --deterministic \
    --require-single-gpu \
    --seed "$seed" \
    --candidate-name "$candidate" \
    --stage full \
    --run-id "$run_id" \
    --base-model-license llama3.1 \
    --base-model-license-url https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE \
    --base-model-source-url "https://huggingface.co/meta-llama/Llama-3.1-8B/tree/${base_revision}" \
    --base-model-card-url https://huggingface.co/meta-llama/Llama-3.1-8B \
    --base-model-license-source-url "https://huggingface.co/meta-llama/Llama-3.1-8B/blob/${base_revision}/README.md" \
    --base-model-metadata-api-url https://huggingface.co/api/models/meta-llama/Llama-3.1-8B \
    --base-model-metadata-checked-at 2026-09-09 \
    --lora-rank 128 \
    --lora-alpha 256 \
    --lora-dropout 0.03 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj \
    --lora-init-adapter "$source_model" \
    --save-adapter-only \
    --source-dataset "$source_description" \
    --source-license "CC BY 4.0" \
    --dataset-attribution-file "${project}/ml/CUAD_ATTRIBUTION.md" \
    --model-id-prefix Llama-3.1-CUAD-transfer \
    --heartbeat-file "$status_file" \
    --telemetry-file "$telemetry_file"
} 2>&1 | tee "$log_file"
