"""Hugging Face legal-encoder classifier training entry point."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import random
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.windowing import (
    DEFAULT_WINDOW_CHARS,
    DEFAULT_WINDOW_STRIDE,
    validate_window_parameters,
)

MODULE_DIR = Path(__file__).resolve().parent
DOMAIN_MAPPING = MODULE_DIR / "cuad_category_domain_mapping.json"
EVALUATION_METRICS_FILE = "evaluation_metrics.json"
WINDOW_CONFIG_FILE = "window_config.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path(os.getenv("SM_CHANNEL_TRAINING", "ml/data/processed")),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("SM_MODEL_DIR", "ml/artifacts/model")),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.getenv("SM_CHECKPOINT_DIR", "ml/artifacts/checkpoints")),
    )
    parser.add_argument(
        "--problem-type",
        choices=("multi_label", "single_label"),
        default="multi_label",
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--positive-weight-cap", type=float, default=1.0)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--evaluation-strategy",
        choices=("epoch", "steps"),
        default="epoch",
    )
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="fp32",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--require-single-gpu", action="store_true")
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-model-license", default=None)
    parser.add_argument("--base-model-license-url", default=None)
    parser.add_argument("--base-model-source-url", default=None)
    parser.add_argument("--base-model-card-url", default=None)
    parser.add_argument("--base-model-license-source-url", default=None)
    parser.add_argument("--base-model-metadata-api-url", default=None)
    parser.add_argument("--base-model-metadata-checked-at", default=None)
    parser.add_argument("--heartbeat-file", type=Path, default=None)
    parser.add_argument("--telemetry-file", type=Path, default=None)
    return parser


def positive_weights(items: list[dict], labels: list[str], cap: float) -> list[float]:
    """Smoothed square-root imbalance weights; training examples only."""
    if not 1 <= cap <= 20:
        raise ValueError("positive-weight-cap must be between 1 and 20")
    counts = {label: sum(label in item["labels"] for item in items) for label in labels}
    return [
        min(cap, max(1.0, ((len(items) - counts[label]) / max(1, counts[label])) ** 0.5))
        for label in labels
    ]


def load_window_config(training_dir: Path) -> dict[str, Any]:
    path = training_dir / WINDOW_CONFIG_FILE
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        config = {
            "schema_version": "1.0",
            "strategy": "answer_centered_positive_sliding_negative",
            "window_chars": DEFAULT_WINDOW_CHARS,
            "stride_chars": DEFAULT_WINDOW_STRIDE,
            "negative_windows_per_positive": 1,
        }
    window_chars = config.get("window_chars")
    stride_chars = config.get("stride_chars")
    if not isinstance(window_chars, int) or not isinstance(stride_chars, int):
        raise ValueError("window_config.json requires integer window and stride values")
    validate_window_parameters(window_chars, stride_chars)
    return config


def load_training_contract(
    training_dir: Path,
    problem_type: str,
    window_chars: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = json.loads((training_dir / "labels.json").read_text(encoding="utf-8"))
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(label, str) and label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("labels.json must contain unique, non-empty strings")
    train = read_jsonl(training_dir / "train.jsonl")
    validation = read_jsonl(training_dir / "validation.jsonl")
    if not train or not validation:
        raise ValueError("training and validation datasets must both be non-empty")
    known_labels = set(labels)
    for split_name, items in (("train", train), ("validation", validation)):
        for item in items:
            if not isinstance(item.get("text"), str) or not item["text"].strip():
                raise ValueError(f"{split_name} record requires non-empty text")
            item_labels = item.get("labels")
            if not isinstance(item_labels, list):
                raise ValueError(f"{split_name} record requires a labels list")
            unknown = set(item_labels) - known_labels
            if unknown:
                raise ValueError(f"{split_name} record has unknown labels: {sorted(unknown)}")
            if problem_type == "single_label" and len(item_labels) != 1:
                raise ValueError("single_label records must contain exactly one label")
            if window_chars is not None and len(item["text"]) > window_chars:
                raise ValueError(f"{split_name} record exceeds configured window_chars")
    return labels, train, validation


def training_api_kwargs(training_arguments_type: type, args: argparse.Namespace) -> dict[str, Any]:
    strategy = getattr(args, "evaluation_strategy", "epoch")
    eval_steps = getattr(args, "eval_steps", 50)
    if strategy == "steps" and eval_steps < 1:
        raise ValueError("eval-steps must be positive for step evaluation")
    if getattr(args, "gradient_accumulation_steps", 1) < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    if getattr(args, "early_stopping_patience", 0) < 0:
        raise ValueError("early-stopping-patience cannot be negative")
    kwargs = {
        "output_dir": str(args.checkpoint_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "save_strategy": strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": ("micro_f1" if args.problem_type == "multi_label" else "accuracy"),
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "data_seed": args.seed,
        "max_steps": getattr(args, "max_steps", -1),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
        "warmup_ratio": getattr(args, "warmup_ratio", 0.0),
        "weight_decay": getattr(args, "weight_decay", 0.0),
        "logging_steps": getattr(args, "logging_steps", 500),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
        "dataloader_num_workers": 0,
        "report_to": [],
    }
    if strategy == "steps":
        kwargs.update({"eval_steps": eval_steps, "save_steps": eval_steps})
    resolved_precision = getattr(args, "resolved_precision", None)
    if resolved_precision in {"fp16", "bf16"}:
        kwargs[resolved_precision] = True
    if getattr(args, "deterministic", False):
        kwargs["full_determinism"] = True
        kwargs["tf32"] = False
    parameters = inspect.signature(training_arguments_type.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[evaluation_key] = strategy
    return kwargs


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def installed_training_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for name in (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return packages


def resolve_precision(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        if requested in {"fp16", "bf16"} and not torch_module.cuda.is_available():
            raise ValueError(f"{requested} precision requires CUDA")
        if requested == "bf16" and not torch_module.cuda.is_bf16_supported():
            raise ValueError("bf16 precision was requested but this GPU does not support it")
        return requested
    if not torch_module.cuda.is_available():
        return "fp32"
    return "bf16" if torch_module.cuda.is_bf16_supported() else "fp16"


def configure_determinism(
    seed: int,
    requested: bool,
    torch_module: Any,
    numpy: Any,
) -> dict[str, Any]:
    random.seed(seed)
    numpy.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if requested:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch_module, "use_deterministic_algorithms"):
            torch_module.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch_module.backends, "cudnn"):
            torch_module.backends.cudnn.benchmark = False
            torch_module.backends.cudnn.deterministic = True
    return {
        "requested": requested,
        "seed": seed,
        "python_hash_seed": os.getenv("PYTHONHASHSEED"),
        "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
            if hasattr(torch_module, "are_deterministic_algorithms_enabled")
            else False
        ),
        "limitations": (
            "Best-effort reproducibility only; GPU hardware, drivers, kernels, package versions, "
            "and distributed execution can still change results."
        ),
    }


def apply_window_config(
    model_config: Any,
    window_config: dict[str, Any],
    tokenizer_max_length: int,
) -> None:
    model_config.contract_window_chars = window_config["window_chars"]
    model_config.contract_window_stride_chars = window_config["stride_chars"]
    model_config.contract_tokenizer_max_length = tokenizer_max_length


def package_artifacts(
    model_dir: Path,
    labels: list[str],
    args: argparse.Namespace,
    metrics: dict[str, Any],
    window_config: dict[str, Any],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    label_to_id = {label: index for index, label in enumerate(labels)}
    mapping = {
        "schema_version": "1.0",
        "label2id": label_to_id,
        "id2label": {str(index): label for label, index in label_to_id.items()},
    }
    requested_revision = getattr(args, "revision", None)
    resolved_revision = getattr(args, "resolved_revision", None)
    config_path = model_dir / "config.json"
    if config_path.is_file() and not resolved_revision:
        resolved_revision = json.loads(config_path.read_text(encoding="utf-8")).get("_commit_hash")
    exact_revision = resolved_revision or requested_revision
    base_model = {
        "model_id": args.model_name,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "exact_revision": exact_revision,
        "revision_is_immutable_sha": bool(
            exact_revision and re.fullmatch(r"[0-9a-fA-F]{40}", exact_revision)
        ),
        "license": getattr(args, "base_model_license", None),
        "license_url": getattr(args, "base_model_license_url", None),
        "source_url": getattr(args, "base_model_source_url", None),
        "model_card_url": getattr(args, "base_model_card_url", None),
        "license_source_url": getattr(args, "base_model_license_source_url", None),
        "metadata_api_url": getattr(args, "base_model_metadata_api_url", None),
        "metadata_checked_at": getattr(args, "base_model_metadata_checked_at", None),
        "license_basis": (
            "Upstream model-card declaration recorded by the matrix; review upstream terms "
            "before redistribution."
        ),
    }
    provenance = {
        "schema_version": "1.0",
        "base_model": args.model_name,
        "base_model_provenance": base_model,
        "model_id": "legal-encoder-prototype",
        "problem_type": args.problem_type,
        "source_dataset": "CUAD v1",
        "source_license": "CC BY 4.0",
        "domain_mapping": DOMAIN_MAPPING.name,
        "labels": labels,
        "metrics_file": EVALUATION_METRICS_FILE,
        "windowing": window_config,
        "tokenizer_max_length": args.max_length,
        "training_options": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "positive_weight_cap": getattr(args, "positive_weight_cap", 1.0),
            "resume_from_checkpoint": getattr(args, "resume_from_checkpoint", None),
            "max_steps": getattr(args, "max_steps", -1),
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", 1),
            "warmup_ratio": getattr(args, "warmup_ratio", 0.0),
            "weight_decay": getattr(args, "weight_decay", 0.0),
            "evaluation_strategy": getattr(args, "evaluation_strategy", "epoch"),
            "early_stopping_patience": getattr(args, "early_stopping_patience", 0),
            "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
            "precision": getattr(args, "resolved_precision", getattr(args, "precision", "auto")),
        },
        "matrix_run": {
            "candidate": getattr(args, "candidate_name", None),
            "stage": getattr(args, "stage", None),
            "run_id": getattr(args, "run_id", None),
        },
        "determinism": getattr(
            args,
            "determinism_metadata",
            {
                "requested": getattr(args, "deterministic", False),
                "seed": args.seed,
                "limitations": (
                    "Seeded execution is not a guarantee of bitwise-identical GPU results."
                ),
            },
        ),
        "runtime": {
            "created_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": installed_training_packages(),
            "accelerator": getattr(args, "accelerator_metadata", {}),
        },
    }
    training_dir = getattr(args, "training_dir", None)
    if training_dir:
        training_dir = Path(training_dir)
        hashes = {
            name: file_sha256(training_dir / f"{name}.jsonl")
            for name in ("train", "validation", "test")
            if (training_dir / f"{name}.jsonl").is_file()
        }
        provenance["split_sha256"] = hashes
        provenance["model_id"] = f"legal-encoder-cuad-{hashes.get('train', 'unknown')[:12]}"
        preparation_path = training_dir / "preparation.json"
        if preparation_path.is_file():
            provenance["preparation_sha256"] = file_sha256(preparation_path)
            shutil.copyfile(preparation_path, model_dir / "training_data_preparation.json")
    weights_path = model_dir / "model.safetensors"
    if weights_path.is_file():
        provenance["weights_sha256"] = file_sha256(weights_path)
        provenance["model_id"] = f"legal-encoder-cuad-{provenance['weights_sha256'][:12]}"
    provenance["base_revision"] = exact_revision
    provenance["requested_revision"] = requested_revision
    (model_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_dir / "base_model_provenance.json").write_text(
        json.dumps(base_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_dir / EVALUATION_METRICS_FILE).write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_dir / WINDOW_CONFIG_FILE).write_text(
        json.dumps(window_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(DOMAIN_MAPPING, model_dir / DOMAIN_MAPPING.name)
    shutil.copyfile(MODULE_DIR / "CUAD_ATTRIBUTION.md", model_dir / "CUAD_ATTRIBUTION.md")
    shutil.copyfile(
        MODULE_DIR / "PRETRAINED_ATTRIBUTION.md", model_dir / "PRETRAINED_ATTRIBUTION.md"
    )
    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for filename in ("inference.py", "heuristic.py", "windowing.py"):
        shutil.copyfile(MODULE_DIR / filename, code_dir / filename)
    artifact_names = (
        "base_model_provenance.json",
        "config.json",
        EVALUATION_METRICS_FILE,
        "label_mapping.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        WINDOW_CONFIG_FILE,
    )
    artifact_hashes = {}
    for name in artifact_names:
        path = model_dir / name
        if not path.is_file():
            continue
        artifact_hashes[name] = (
            provenance["weights_sha256"]
            if name == "model.safetensors"
            else file_sha256(path)
        )
    provenance["artifact_sha256"] = artifact_hashes
    (model_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max-steps must be -1 or a positive integer")
    if args.deterministic:
        # This must be present before CUDA is initialized. The matrix also sets it
        # in the child environment so resumed subprocesses have the same setting.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.heartbeat_file:
        atomic_json(
            args.heartbeat_file,
            {
                "state": "loading",
                "run_id": args.run_id,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
    window_config = load_window_config(args.training_dir)
    labels, train_items, validation_items = load_training_contract(
        args.training_dir,
        args.problem_type,
        window_config["window_chars"],
    )
    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise SystemExit(
            "training requires packages from ml/requirements-training.txt "
            "or ml/requirements-training-gpu.txt"
        ) from error

    set_seed(args.seed)
    args.determinism_metadata = configure_determinism(args.seed, args.deterministic, torch, np)
    if args.require_single_gpu and torch.cuda.device_count() != 1:
        raise SystemExit(
            "one-GPU matrix invariant violated: exactly one CUDA device must be visible, "
            f"found {torch.cuda.device_count()}"
        )
    args.resolved_precision = resolve_precision(torch, args.precision)
    args.accelerator_metadata = {
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": torch.cuda.device_count(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)

    class ContractDataset(Dataset):
        def __init__(self, items: list[dict[str, Any]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.items[index]
            encoded = tokenizer(item["text"], truncation=True, max_length=args.max_length)
            if args.problem_type == "multi_label":
                target = [0.0] * len(labels)
                for label in item["labels"]:
                    target[label_to_id[label]] = 1.0
                encoded["labels"] = target
            else:
                encoded["labels"] = label_to_id[item["labels"][0]]
            return encoded

    def compute_metrics(result: Any) -> dict[str, float]:
        logits, references = result
        if args.problem_type == "multi_label":
            predictions = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
            references = references.astype(int)
            tp = int(np.logical_and(predictions == 1, references == 1).sum())
            fp = int(np.logical_and(predictions == 1, references == 0).sum())
            fn = int(np.logical_and(predictions == 0, references == 1).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            exact = float(np.all(predictions == references, axis=1).mean())
            return {
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
                "exact_match": exact,
            }
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == references).mean())}

    problem_type = (
        "multi_label_classification"
        if args.problem_type == "multi_label"
        else "single_label_classification"
    )
    model_config = AutoConfig.from_pretrained(
        args.model_name,
        revision=args.revision,
    )
    # ModernBERT enables internal torch.compile paths whenever Triton is
    # present. CUDA 13 identifies B300 as sm_103a, which the bundled ptxas
    # cannot compile yet; eager reference layers are deterministic and avoid
    # silently dropping this candidate from the matrix.
    if hasattr(model_config, "reference_compile"):
        model_config.reference_compile = False
    model_config.num_labels = len(labels)
    model_config.id2label = {index: label for label, index in label_to_id.items()}
    model_config.label2id = label_to_id
    model_config.problem_type = problem_type
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.revision,
        config=model_config,
    )
    args.resolved_revision = (
        getattr(model.config, "_commit_hash", None)
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    )
    if (
        args.revision
        and re.fullmatch(r"[0-9a-fA-F]{40}", args.revision)
        and args.resolved_revision
        and args.revision.lower() != args.resolved_revision.lower()
    ):
        raise RuntimeError(
            f"resolved model revision {args.resolved_revision} does not match {args.revision}"
        )
    apply_window_config(model.config, window_config, args.max_length)
    training_arguments = TrainingArguments(**training_api_kwargs(TrainingArguments, args))
    weights = positive_weights(train_items, labels, args.positive_weight_cap)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            targets = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.logits,
                targets.to(outputs.logits.dtype),
                pos_weight=torch.tensor(
                    weights, device=outputs.logits.device, dtype=outputs.logits.dtype
                ),
            )
            return (loss, outputs) if return_outputs else loss

    class Progress(TrainerCallback):
        def __init__(self):
            self.last_heartbeat = 0.0

        def record(self, event, state, **values):
            if not args.heartbeat_file and not args.telemetry_file:
                return
            now = time.monotonic()
            heartbeat_due = (
                event != "step"
                or now - self.last_heartbeat >= 30
                or state.global_step == state.max_steps
            )
            row = {
                "event": event,
                "run_id": args.run_id,
                "candidate": args.candidate_name,
                "stage": args.stage,
                "step": state.global_step,
                "total_steps": state.max_steps,
                "epoch": state.epoch,
                "updated_at": datetime.now(UTC).isoformat(),
                **values,
            }
            if heartbeat_due and args.heartbeat_file:
                atomic_json(args.heartbeat_file, {"state": "training", **row})
                self.last_heartbeat = now
            if event != "step" and args.telemetry_file:
                args.telemetry_file.parent.mkdir(parents=True, exist_ok=True)
                with args.telemetry_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")

        def on_train_begin(self, args_, state, control, **kwargs):
            self.record("train_begin", state)

        def on_step_end(self, args_, state, control, **kwargs):
            self.record("step", state)
            if state.global_step % 50 == 0 or state.global_step == state.max_steps:
                path = args.checkpoint_dir / "progress.json"
                atomic_json(
                    path,
                    {
                        "step": state.global_step,
                        "total": state.max_steps,
                        "epoch": state.epoch,
                        "run_id": args.run_id,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                )

        def on_log(self, args_, state, control, logs=None, **kwargs):
            self.record("log", state, metrics=logs or {})

        def on_evaluate(self, args_, state, control, metrics=None, **kwargs):
            self.record("validation", state, metrics=metrics or {})

        def on_train_end(self, args_, state, control, **kwargs):
            self.record("train_end", state)

    trainer_type = (
        WeightedTrainer
        if args.positive_weight_cap > 1 and args.problem_type == "multi_label"
        else Trainer
    )
    callbacks: list[Any] = [Progress()]
    if args.early_stopping_patience:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
    trainer = trainer_type(
        model=model,
        args=training_arguments,
        train_dataset=ContractDataset(train_items),
        eval_dataset=ContractDataset(validation_items),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(str(args.model_dir))
    package_artifacts(args.model_dir, labels, args, metrics, window_config)
    summary = {
        "state": "complete",
        "run_id": args.run_id,
        "candidate": args.candidate_name,
        "stage": args.stage,
        "seed": args.seed,
        "metrics": metrics,
        "model_dir": str(args.model_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_json(args.checkpoint_dir / "run-summary.json", summary)
    if args.heartbeat_file:
        atomic_json(args.heartbeat_file, summary)


if __name__ == "__main__":
    main()
