#!/usr/bin/env bash
set -euo pipefail

submit=false
if [[ "${1:-}" == "--submit" ]]; then
  submit=true
elif [[ -n "${1:-}" ]]; then
  echo "Usage: $0 [--submit]" >&2
  exit 64
fi

: "${SAGEMAKER_TRAINING_BUCKET:?Set SAGEMAKER_TRAINING_BUCKET from the Terraform output.}"
: "${SAGEMAKER_ROLE_ARN:?Set SAGEMAKER_ROLE_ARN from the Terraform output.}"

AWS_REGION="${AWS_REGION:-us-gov-west-1}"
MODEL_FAMILY="${MODEL_FAMILY:-legal-bert}"
SAGEMAKER_CONFIG="${SAGEMAKER_CONFIG:-ml/configs/sagemaker.yaml}"
TRAINING_DIR="${TRAINING_DIR:-ml/data/processed}"
[[ "$MODEL_FAMILY" =~ ^[a-z0-9][a-z0-9-]{1,62}$ ]] || {
  echo "MODEL_FAMILY must contain lowercase letters, digits, and hyphens." >&2
  exit 64
}
[[ -s "$SAGEMAKER_CONFIG" ]] || {
  echo "Missing SageMaker model configuration: ${SAGEMAKER_CONFIG}" >&2
  exit 66
}
TRAINING_URI="s3://${SAGEMAKER_TRAINING_BUCKET}/input/${MODEL_FAMILY}"
OUTPUT_URI="s3://${SAGEMAKER_TRAINING_BUCKET}/output/${MODEL_FAMILY}"
CHECKPOINT_URI="s3://${SAGEMAKER_TRAINING_BUCKET}/checkpoints/${MODEL_FAMILY}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  echo "Python 3 is required." >&2
  exit 69
fi

for file in labels.json train.jsonl validation.jsonl; do
  [[ -s "${TRAINING_DIR}/${file}" ]] || {
    echo "Missing required training file: ${TRAINING_DIR}/${file}" >&2
    exit 66
  }
done

identity_arn="$(aws sts get-caller-identity --query Arn --output text)"
[[ "$identity_arn" == arn:aws-us-gov:* ]] || {
  echo "The active AWS identity is not in GovCloud: ${identity_arn}" >&2
  exit 77
}

aws s3 sync "$TRAINING_DIR" "$TRAINING_URI" \
  --region "$AWS_REGION" \
  --exclude "*" \
  --include "labels.json" \
  --include "train.jsonl" \
  --include "validation.jsonl" \
  --include "window_config.json" \
  --only-show-errors

launcher_args=(
  --config "$SAGEMAKER_CONFIG"
  --role-arn "$SAGEMAKER_ROLE_ARN"
  --training-s3-uri "$TRAINING_URI"
  --output-s3-uri "$OUTPUT_URI"
  --checkpoint-s3-uri "$CHECKPOINT_URI"
)

if [[ "$submit" == true ]]; then
  echo "Submitting a billable SageMaker training job for ${MODEL_FAMILY}."
  launcher_args+=(--submit)
else
  echo "Dry run only; rerun with --submit to start billable GPU training."
fi

"$PYTHON_BIN" -m ml.launch_sagemaker "${launcher_args[@]}"
