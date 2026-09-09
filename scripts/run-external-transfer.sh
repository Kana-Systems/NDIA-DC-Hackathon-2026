#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <maud-llama|maud-caselaw|contract-llama-4k|contract-llama-8k> <smoke|full>" >&2
  exit 2
fi

branch="$1"
mode="$2"
project="${PROJECT_ROOT:-/workspace/project-external-20260909}"
venv="${VENV_ROOT:-/workspace/venv}"
python="${venv}/bin/python"
base_revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
leader="${project}/artifacts/models/ndia-cuad-llama-best-20260909T1205Z/full--Llama-3.1-8B-r128-lr075-cap30-d03--seed-17"
run_root="/workspace/external-transfer-20260909"
offline=1

if [[ ! -x "$python" ]]; then
  echo "training Python is missing: $python" >&2
  exit 1
fi
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
  echo "mode must be smoke or full" >&2
  exit 2
fi

case "$branch" in
  maud-llama)
    training_dir="${project}/ml/data/external/processed/maud"
    model_dir="${run_root}/models/maud-llama-r128-seed17-${mode}"
    checkpoint_dir="${run_root}/checkpoints/maud-llama-r128-seed17-${mode}"
    run_id="maud-llama-r128-seed17-${mode}"
    command=(
      "$python" -m ml.train
      --model-name meta-llama/Llama-3.1-8B
      --revision "$base_revision"
      --training-dir "$training_dir"
      --model-dir "$model_dir"
      --checkpoint-dir "$checkpoint_dir"
      --problem-type single_label
      --epochs 3
      --batch-size 8
      --gradient-accumulation-steps 4
      --learning-rate 5e-5
      --max-length 1024
      --warmup-ratio 0.1
      --weight-decay 0.01
      --metric-for-best-model macro_f1
      --evaluation-strategy steps
      --eval-steps 200
      --logging-steps 20
      --early-stopping-patience 2
      --precision bf16
      --gradient-checkpointing
      --deterministic
      --require-single-gpu
      --seed 17
      --candidate-name maud-llama-r128
      --stage "$mode"
      --run-id "$run_id"
      --base-model-license llama3.1
      --base-model-license-url https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
      --base-model-source-url "https://huggingface.co/meta-llama/Llama-3.1-8B/tree/${base_revision}"
      --base-model-card-url https://huggingface.co/meta-llama/Llama-3.1-8B
      --base-model-license-source-url "https://huggingface.co/meta-llama/Llama-3.1-8B/blob/${base_revision}/README.md"
      --base-model-metadata-api-url https://huggingface.co/api/models/meta-llama/Llama-3.1-8B
      --base-model-metadata-checked-at 2026-09-09
      --lora-rank 128
      --lora-alpha 256
      --lora-dropout 0.03
      --lora-target-modules q_proj,k_proj,v_proj,o_proj
      --lora-init-adapter "$leader"
      --save-adapter-only
      --source-dataset "MAUD v1 publisher-train-derived development split"
      --source-license "CC BY 4.0"
      --dataset-attribution-file "${project}/ml/EXTERNAL_LEGAL_DATA_ATTRIBUTION.md"
      --model-id-prefix Llama-3.1-MAUD
    )
    ;;
  maud-caselaw)
    offline=0
    training_dir="${project}/ml/data/external/processed/maud"
    model_dir="${run_root}/models/maud-caselaw-seed29-${mode}"
    checkpoint_dir="${run_root}/checkpoints/maud-caselaw-seed29-${mode}"
    run_id="maud-caselaw-seed29-${mode}"
    command=(
      "$python" -m ml.train
      --model-name ai-law-society-lab/CaseLawModernBERT-large
      --revision ae93d346c3037d15193c4d0266cda2cae50fec09
      --training-dir "$training_dir"
      --model-dir "$model_dir"
      --checkpoint-dir "$checkpoint_dir"
      --problem-type single_label
      --epochs 3
      --batch-size 32
      --learning-rate 1e-5
      --max-length 512
      --warmup-ratio 0.1
      --weight-decay 0.01
      --metric-for-best-model macro_f1
      --evaluation-strategy steps
      --eval-steps 200
      --logging-steps 20
      --early-stopping-patience 2
      --precision bf16
      --deterministic
      --require-single-gpu
      --seed 29
      --candidate-name maud-caselaw-modernbert
      --stage "$mode"
      --run-id "$run_id"
      --base-model-license apache-2.0
      --base-model-license-url https://www.apache.org/licenses/LICENSE-2.0
      --base-model-source-url https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large/tree/ae93d346c3037d15193c4d0266cda2cae50fec09
      --base-model-card-url https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large
      --base-model-license-source-url https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large/blob/ae93d346c3037d15193c4d0266cda2cae50fec09/README.md
      --base-model-metadata-api-url https://huggingface.co/api/models/ai-law-society-lab/CaseLawModernBERT-large
      --base-model-metadata-checked-at 2026-09-09
      --source-dataset "MAUD v1 publisher-train-derived development split"
      --source-license "CC BY 4.0"
      --dataset-attribution-file "${project}/ml/EXTERNAL_LEGAL_DATA_ATTRIBUTION.md"
      --model-id-prefix CaseLawModernBERT-MAUD
    )
    ;;
  contract-llama-4k|contract-llama-8k)
    training_dir="${project}/ml/data/external/processed/contractnli"
    if [[ "$branch" == "contract-llama-4k" ]]; then
      max_length=4096
      batch_size=4
      accumulation=8
      seed=17
      epochs=3
    else
      max_length=8192
      batch_size=2
      accumulation=16
      seed=29
      epochs=2
    fi
    model_dir="${run_root}/models/${branch}-r128-seed${seed}-${mode}"
    checkpoint_dir="${run_root}/checkpoints/${branch}-r128-seed${seed}-${mode}"
    run_id="${branch}-r128-seed${seed}-${mode}"
    command=(
      "$python" -m ml.train
      --model-name meta-llama/Llama-3.1-8B
      --revision "$base_revision"
      --training-dir "$training_dir"
      --model-dir "$model_dir"
      --checkpoint-dir "$checkpoint_dir"
      --problem-type single_label
      --epochs "$epochs"
      --batch-size "$batch_size"
      --gradient-accumulation-steps "$accumulation"
      --learning-rate 5e-5
      --max-length "$max_length"
      --warmup-ratio 0.1
      --weight-decay 0.01
      --metric-for-best-model macro_f1
      --evaluation-strategy steps
      --eval-steps 100
      --logging-steps 10
      --early-stopping-patience 2
      --precision bf16
      --gradient-checkpointing
      --deterministic
      --require-single-gpu
      --seed "$seed"
      --candidate-name "${branch}-r128"
      --stage "$mode"
      --run-id "$run_id"
      --base-model-license llama3.1
      --base-model-license-url https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
      --base-model-source-url "https://huggingface.co/meta-llama/Llama-3.1-8B/tree/${base_revision}"
      --base-model-card-url https://huggingface.co/meta-llama/Llama-3.1-8B
      --base-model-license-source-url "https://huggingface.co/meta-llama/Llama-3.1-8B/blob/${base_revision}/README.md"
      --base-model-metadata-api-url https://huggingface.co/api/models/meta-llama/Llama-3.1-8B
      --base-model-metadata-checked-at 2026-09-09
      --lora-rank 128
      --lora-alpha 256
      --lora-dropout 0.03
      --lora-target-modules q_proj,k_proj,v_proj,o_proj
      --lora-init-adapter "$leader"
      --save-adapter-only
      --source-dataset "ContractNLI publisher-train-derived development split"
      --source-license "CC BY 4.0"
      --dataset-attribution-file "${project}/ml/EXTERNAL_LEGAL_DATA_ATTRIBUTION.md"
      --model-id-prefix Llama-3.1-ContractNLI
    )
    ;;
  *)
    echo "unknown branch: $branch" >&2
    exit 2
    ;;
esac

if [[ "$mode" == "smoke" ]]; then
  command+=(--max-steps 12)
fi

mkdir -p "$model_dir" "$checkpoint_dir" "${run_root}/logs" "${run_root}/status"
log="${run_root}/logs/${run_id}.log"
status="${run_root}/status/${run_id}.json"
pid_file="${run_root}/status/${run_id}.pid"
if [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo "run is already active: $run_id" >&2
  exit 1
fi

export PYTHONPATH="$project"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/hf-hub-cache-llama31}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-$offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-$offline}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED
PYTHONHASHSEED="$(printf '%s' "$run_id" | cksum | awk '{print $1}')"

nohup "${command[@]}" \
  --heartbeat-file "$status" \
  --telemetry-file "${run_root}/status/${run_id}.jsonl" \
  >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pid_file"
printf 'started %s pid=%s log=%s\n' "$run_id" "$pid" "$log"
