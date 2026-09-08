# Contract classifier prototype

The production-oriented path fine-tunes a Hugging Face legal encoder for
sequence classification. The offline path is a deterministic heuristic baseline;
tests and the sample benchmark never download a model.

## Data and training

1. Review `CUAD_ATTRIBUTION.md`, then optionally run
   `python -m ml.download_cuad --accept-license --sha256 EXPECTED_DIGEST`.
2. Run `python -m ml.preprocess_cuad ml/data/cuad.zip`. This creates
   document-grouped deterministic `train.jsonl`, `validation.jsonl`, and
   `labels.json`; grouping prevents any window from one contract crossing splits.
   Positive examples are bounded, answer-centered windows built from CUAD's
   `answer_start` and answer text spans. Nearby answer spans contribute all
   applicable labels to an overlapping window. Deterministic answer-free sliding
   windows supply negative examples; an entire contract is never assigned one
   label set and truncated.
3. Install `requirements-training.txt` and run `python -m ml.train`. Override
   `--model-name` with a local model directory for disconnected training.

Training input records use `{"id", "document_id", "text", "labels"}`. Labels
are strings listed in `labels.json`. The default multi-label mode creates
independent sigmoid outputs; `--problem-type single_label` uses softmax.
Each record also carries source character bounds and `is_negative`.
`window_config.json` records the preprocessing strategy and defaults:
1,800-character windows, 900-character stride, and one negative per positive.
Override these with `--window-chars`, `--stride-chars`, and
`--negative-windows-per-positive`.
`cuad_category_domain_mapping.json` provides an app-readable, explicitly
advisory mapping from selected CUAD categories to review domains. CUAD is a
commercial-contract dataset: neither its labels nor mapped domains can establish
FAR or DFARS applicability.

Artifacts include model/tokenizer files, the model `config.json` id2label data,
`label_mapping.json`, `training_provenance.json`, evaluation metrics, CUAD
attribution/domain mapping, and deployable `code/inference.py` plus
`code/heuristic.py` and `code/windowing.py`. Window settings are retained in
`window_config.json`, training provenance, and custom model `config.json`
properties. Checkpoints use `SM_CHECKPOINT_DIR`, keeping intermediate training
state out of the final SageMaker model archive.

## SageMaker

`configs/sagemaker.yaml` defines a tagged single-GPU `ml.g6.xlarge` Hugging Face
job for GovCloud. Terraform provisions its encrypted bucket, execution role, and
an optional submitter policy. From the repository root, install
`requirements-sagemaker.txt`, set the Terraform output values, and run:

```bash
export SAGEMAKER_TRAINING_BUCKET=BUCKET
export SAGEMAKER_ROLE_ARN=ARN
./scripts/train-sagemaker.sh
```

The script validates and uploads the processed files. It is dry-run by default;
add `--submit` to create a billable job.
SageMaker supplies the training channel and model output paths through
`SM_CHANNEL_TRAINING` and `SM_MODEL_DIR`. The launcher resolves `source_dir`
from its own location, so invocation does not depend on the current directory.
The training entry point detects both current `eval_strategy` and older
`evaluation_strategy` Transformers APIs.

## Inference contract

`inference.py` exposes SageMaker `model_fn`, `input_fn`, `predict_fn`, and
`output_fn`. Requests are JSON:

`{"text": "...", "threshold": 0.5}` or `{"texts": ["...", "..."]}`.

Responses are `{"predictions": [{"model_id": "...", "labels":
[{"label": "...", "score": 0.9}]}]}`. A model directory without a transformer
`config.json` selects `local-heuristic-v1`. Package `inference.py`,
`heuristic.py`, `windowing.py`, and model files in the serving artifact's `code/`
layout. Trained-transformer inference applies the persisted sliding windows to
each original input and returns one prediction per input, taking each label's
maximum score across its windows. The heuristic fallback remains a single-pass,
deterministic baseline.

`model_manifest.yaml` records provenance, intended use, and limitations.
Classifier output is decision support, not an applicability or legal conclusion.
