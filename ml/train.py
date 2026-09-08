"""Hugging Face legal-encoder classifier training entry point."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any

from ml.windowing import (
    DEFAULT_WINDOW_CHARS,
    DEFAULT_WINDOW_STRIDE,
    validate_window_parameters,
)

MODULE_DIR = Path(__file__).resolve().parent
DOMAIN_MAPPING = MODULE_DIR / "cuad_category_domain_mapping.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
    path = training_dir / "window_config.json"
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
    kwargs = {
        "output_dir": str(args.checkpoint_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": ("micro_f1" if args.problem_type == "multi_label" else "accuracy"),
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "report_to": [],
    }
    parameters = inspect.signature(training_arguments_type.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[evaluation_key] = "epoch"
    return kwargs


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
    label_to_id = {label: index for index, label in enumerate(labels)}
    mapping = {
        "schema_version": "1.0",
        "label2id": label_to_id,
        "id2label": {str(index): label for label, index in label_to_id.items()},
    }
    provenance = {
        "schema_version": "1.0",
        "base_model": args.model_name,
        "model_id": "legal-encoder-prototype",
        "problem_type": args.problem_type,
        "source_dataset": "CUAD v1",
        "source_license": "CC BY 4.0",
        "domain_mapping": DOMAIN_MAPPING.name,
        "labels": labels,
        "metrics_file": "evaluation_metrics.json",
        "windowing": window_config,
        "tokenizer_max_length": args.max_length,
        "training_options": {
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "positive_weight_cap": getattr(args, "positive_weight_cap", 1.0),
            "resume_from_checkpoint": getattr(args, "resume_from_checkpoint", None),
        },
    }
    training_dir = getattr(args, "training_dir", None)
    if training_dir:
        hashes = {
            name: hashlib.sha256((Path(training_dir) / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in ("train", "validation", "test")
            if (Path(training_dir) / f"{name}.jsonl").is_file()
        }
        provenance["split_sha256"] = hashes
        provenance["model_id"] = f"legal-encoder-cuad-{hashes.get('train', 'unknown')[:12]}"
    weights_path = model_dir / "model.safetensors"
    if weights_path.is_file():
        provenance["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        provenance["model_id"] = f"legal-encoder-cuad-{provenance['weights_sha256'][:12]}"
    config_path = model_dir / "config.json"
    if config_path.is_file():
        provenance["base_revision"] = json.loads(config_path.read_text()).get("_commit_hash")
    provenance["requested_revision"] = getattr(args, "revision", None)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (model_dir / "window_config.json").write_text(
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


def main() -> None:
    args = build_parser().parse_args()
    window_config = load_window_config(args.training_dir)
    labels, train_items, validation_items = load_training_contract(
        args.training_dir,
        args.problem_type,
        window_config["window_chars"],
    )
    try:
        import numpy as np
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise SystemExit("training requires packages from ml/requirements-training.txt") from error

    set_seed(args.seed)
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
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        revision=args.revision,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
        problem_type=problem_type,
    )
    apply_window_config(model.config, window_config, args.max_length)
    training_arguments = TrainingArguments(**training_api_kwargs(TrainingArguments, args))
    weights = positive_weights(train_items, labels, args.positive_weight_cap)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch

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
        def on_step_end(self, args_, state, control, **kwargs):
            if state.global_step % 50 == 0 or state.global_step == state.max_steps:
                path = args.checkpoint_dir / "progress.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".pending.json")
                temp.write_text(
                    json.dumps(
                        {"step": state.global_step, "total": state.max_steps, "epoch": state.epoch}
                    )
                )
                temp.replace(path)

    trainer_type = (
        WeightedTrainer
        if args.positive_weight_cap > 1 and args.problem_type == "multi_label"
        else Trainer
    )
    trainer = trainer_type(
        model=model,
        args=training_arguments,
        train_dataset=ContractDataset(train_items),
        eval_dataset=ContractDataset(validation_items),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[Progress()],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model(str(args.model_dir))
    tokenizer.save_pretrained(str(args.model_dir))
    package_artifacts(args.model_dir, labels, args, metrics, window_config)


if __name__ == "__main__":
    main()
