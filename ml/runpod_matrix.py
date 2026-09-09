"""Sequential, resumable one-GPU candidate training matrix.

This module only trains and ranks on validation metrics. It never reads test
labels for selection and never changes the selected/deployed model registry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ml.improvement_data import prepare

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0"
DEFAULT_SEEDS = (17, 29, 43)
MINIMUM_RESERVE_SECONDS = 2 * 60 * 60
PINNED_SHA = re.compile(r"^[0-9a-f]{40}$")
TERMINAL_RUN_STATES = frozenset({"complete", "failed"})
RESUMABLE_RUN_STATES = frozenset(
    {"pending", "planned", "running", "interrupted", "deadline_paused"}
)

PINNED_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "legal-bert-weighted",
        "model_name": "nlpaueb/legal-bert-base-uncased",
        "revision": "15b570cbf88259610b082a167dacc190124f60f6",
        "license": "cc-by-sa-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "model_card_url": "https://huggingface.co/nlpaueb/legal-bert-base-uncased",
        "metadata_api_url": ("https://huggingface.co/api/models/nlpaueb/legal-bert-base-uncased"),
        "metadata_checked_at": "2026-09-08",
        "source_url": (
            "https://huggingface.co/nlpaueb/legal-bert-base-uncased/tree/"
            "15b570cbf88259610b082a167dacc190124f60f6"
        ),
        "license_source_url": (
            "https://huggingface.co/nlpaueb/legal-bert-base-uncased/blob/"
            "15b570cbf88259610b082a167dacc190124f60f6/README.md"
        ),
        "positive_weight_cap": 4.0,
        "learning_rate": 2e-5,
        "batch_size": 64,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "max_length": 512,
    },
    {
        "name": "deberta-v3-large",
        "model_name": "microsoft/deberta-v3-large",
        "revision": "64a8c8eab3e352a784c658aef62be1662607476f",
        "license": "mit",
        "license_url": "https://opensource.org/license/mit/",
        "model_card_url": "https://huggingface.co/microsoft/deberta-v3-large",
        "metadata_api_url": "https://huggingface.co/api/models/microsoft/deberta-v3-large",
        "metadata_checked_at": "2026-09-08",
        "source_url": (
            "https://huggingface.co/microsoft/deberta-v3-large/tree/"
            "64a8c8eab3e352a784c658aef62be1662607476f"
        ),
        "license_source_url": (
            "https://huggingface.co/microsoft/deberta-v3-large/blob/"
            "64a8c8eab3e352a784c658aef62be1662607476f/README.md"
        ),
        "positive_weight_cap": 1.0,
        "learning_rate": 1e-5,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "max_length": 512,
    },
    {
        "name": "caselaw-modernbert-large",
        "model_name": "ai-law-society-lab/CaseLawModernBERT-large",
        "revision": "ae93d346c3037d15193c4d0266cda2cae50fec09",
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "model_card_url": ("https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large"),
        "metadata_api_url": (
            "https://huggingface.co/api/models/ai-law-society-lab/CaseLawModernBERT-large"
        ),
        "metadata_checked_at": "2026-09-08",
        "source_url": (
            "https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large/tree/"
            "ae93d346c3037d15193c4d0266cda2cae50fec09"
        ),
        "license_source_url": (
            "https://huggingface.co/ai-law-society-lab/CaseLawModernBERT-large/blob/"
            "ae93d346c3037d15193c4d0266cda2cae50fec09/README.md"
        ),
        "positive_weight_cap": 1.0,
        "learning_rate": 1e-5,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "max_length": 512,
    },
    {
        "name": "roberta-large",
        "model_name": "FacebookAI/roberta-large",
        "revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
        "license": "mit",
        "license_url": "https://opensource.org/license/mit/",
        "model_card_url": "https://huggingface.co/FacebookAI/roberta-large",
        "metadata_api_url": "https://huggingface.co/api/models/FacebookAI/roberta-large",
        "metadata_checked_at": "2026-09-08",
        "source_url": (
            "https://huggingface.co/FacebookAI/roberta-large/tree/"
            "722cf37b1afa9454edce342e7895e588b6ff1d59"
        ),
        "license_source_url": (
            "https://huggingface.co/FacebookAI/roberta-large/blob/"
            "722cf37b1afa9454edce342e7895e588b6ff1d59/README.md"
        ),
        "positive_weight_cap": 1.0,
        "learning_rate": 1e-5,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "max_length": 512,
    },
)
CANDIDATES = PINNED_CANDIDATES
SEEDS = DEFAULT_SEEDS
DEADLINE_RESERVE_SECONDS = MINIMUM_RESERVE_SECONDS


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    candidate: str
    stage: str
    seed: int
    epochs: float
    max_steps: int
    evaluation_strategy: str
    eval_steps: int
    parent_run_id: str | None = None
    gpus: int = 1


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    timed_out: bool
    duration_seconds: float


def default_config() -> dict[str, Any]:
    """Return a mutable copy of the reviewed default matrix."""
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_metric": "eval_micro_f1",
        "finalist_count": 2,
        "seeds": list(DEFAULT_SEEDS),
        "execution": {
            "gpus_per_run": 1,
            "max_concurrent_runs": 1,
            "deterministic": True,
            "precision": "auto",
        },
        "candidates": copy.deepcopy(list(PINNED_CANDIDATES)),
        "smoke": {
            "name": "smoke",
            "epochs": 1.0,
            "max_steps": 12,
            "eval_steps": 6,
            "keep_fraction": 1.0,
            "early_stopping_patience": 1,
        },
        "screens": [
            {
                "name": "screen-1",
                "epochs": 1.0,
                "max_steps": 96,
                "eval_steps": 24,
                "keep_fraction": 0.5,
                "early_stopping_patience": 1,
            },
            {
                "name": "screen-2",
                "epochs": 1.0,
                "max_steps": 288,
                "eval_steps": 48,
                "keep_count": 2,
                "early_stopping_patience": 1,
            },
        ],
        "full": {
            "name": "full",
            "epochs": 3.0,
            "max_steps": -1,
            "eval_steps": 50,
            "early_stopping_patience": 2,
        },
        "optimizer": {"warmup_ratio": 0.1, "weight_decay": 0.01},
    }


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_matrix_config(path: Path | None = None) -> dict[str, Any]:
    config = default_config()
    if path is not None:
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ValueError("matrix config must be a JSON object")
        config = _deep_merge(config, override)
    return validate_matrix_config(config)


def _validate_execution(execution: Mapping[str, Any]) -> None:
    if execution.get("gpus_per_run") != 1 or execution.get("max_concurrent_runs") != 1:
        raise ValueError("the matrix requires one GPU and exactly one concurrent run")
    if execution.get("precision") not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("execution.precision must be auto, fp32, fp16, or bf16")
    if not isinstance(execution.get("deterministic"), bool):
        raise ValueError("execution.deterministic must be boolean")


def _validate_candidate(candidate: Any) -> str:
    required_fields = (
        "name",
        "model_name",
        "revision",
        "license",
        "license_url",
        "model_card_url",
        "metadata_api_url",
        "metadata_checked_at",
        "source_url",
        "license_source_url",
    )
    if not isinstance(candidate, dict):
        raise ValueError("candidate entries must be objects")
    missing = [key for key in required_fields if not candidate.get(key)]
    if missing:
        raise ValueError(f"candidate is missing fields: {', '.join(missing)}")
    name = str(candidate["name"])
    if not PINNED_SHA.fullmatch(str(candidate["revision"]).lower()):
        raise ValueError(f"{name} revision must be a full 40-character SHA")
    for field in ("batch_size", "gradient_accumulation_steps", "max_length"):
        value = candidate.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} {field} must be positive")
    if float(candidate.get("learning_rate", 0)) <= 0:
        raise ValueError(f"{name} learning_rate must be positive")
    if not 1 <= float(candidate.get("positive_weight_cap", 0)) <= 20:
        raise ValueError(f"{name} positive_weight_cap must be in [1, 20]")
    if not isinstance(candidate.get("gradient_checkpointing"), bool):
        raise ValueError(f"{name} gradient_checkpointing must be boolean")
    extra_args = candidate.get("extra_args", [])
    if extra_args is None:
        extra_args = []
        candidate["extra_args"] = extra_args
    if not isinstance(extra_args, list) or not all(
        isinstance(item, str) and item.strip() for item in extra_args
    ):
        raise ValueError(f"{name} extra_args must be a list of non-empty strings")
    return name


def _validate_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("at least two candidate configurations are required")
    names = [_validate_candidate(candidate) for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    return candidates


def _validate_seeds(seeds: Any) -> list[int]:
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty unique list")
    if not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be integers")
    return seeds


def _validate_stage(stage: Any) -> str:
    if not isinstance(stage, dict) or not stage.get("name"):
        raise ValueError("smoke, screens, and full stage configurations are required")
    name = str(stage["name"])
    max_steps = stage.get("max_steps")
    if not isinstance(max_steps, int) or max_steps == 0 or max_steps < -1:
        raise ValueError(f"{name} max_steps must be -1 or positive")
    if float(stage.get("epochs", 0)) <= 0:
        raise ValueError(f"{name} epochs must be positive")
    eval_steps = stage.get("eval_steps")
    if not isinstance(eval_steps, int) or isinstance(eval_steps, bool) or eval_steps <= 0:
        raise ValueError(f"{name} eval_steps must be positive")
    if int(stage.get("early_stopping_patience", -1)) < 0:
        raise ValueError(f"{name} early_stopping_patience cannot be negative")
    return name


def _validate_stages(config: Mapping[str, Any]) -> None:
    screens = config.get("screens")
    if not isinstance(screens, list) or not screens:
        raise ValueError("at least one successive-halving screen is required")
    stages = [config.get("smoke"), *screens, config.get("full")]
    names = [_validate_stage(stage) for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("stage names must be unique")


def _validate_optimizer(optimizer: Mapping[str, Any]) -> None:
    if not 0 <= float(optimizer.get("warmup_ratio", -1)) < 1:
        raise ValueError("optimizer.warmup_ratio must be in [0, 1)")
    if float(optimizer.get("weight_decay", -1)) < 0:
        raise ValueError("optimizer.weight_decay cannot be negative")


def validate_matrix_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(config))
    _validate_execution(normalized.get("execution", {}))
    metric = normalized.get("selection_metric")
    if not isinstance(metric, str) or not metric.startswith("eval_") or "test" in metric.lower():
        raise ValueError("selection_metric must be a validation eval_* metric")
    candidates = _validate_candidates(normalized.get("candidates"))
    _validate_seeds(normalized.get("seeds"))
    finalist_count = normalized.get("finalist_count")
    if not isinstance(finalist_count, int) or not 1 <= finalist_count <= len(candidates):
        raise ValueError("finalist_count is outside the candidate range")
    _validate_stages(normalized)
    _validate_optimizer(normalized.get("optimizer", {}))
    return normalized


def candidate_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["name"]: candidate for candidate in config["candidates"]}


def run_id(stage: str, candidate: str, seed: int) -> str:
    safe_stage = re.sub(r"[^a-zA-Z0-9_.-]+", "-", stage)
    safe_candidate = re.sub(r"[^a-zA-Z0-9_.-]+", "-", candidate)
    return f"{safe_stage}--{safe_candidate}--seed-{seed}"


def runs_for_stage(
    config: Mapping[str, Any],
    stage: Mapping[str, Any],
    candidates: Sequence[str],
    parent_stage: str | None = None,
) -> list[RunSpec]:
    seed = int(config["seeds"][0])
    return [
        RunSpec(
            run_id=run_id(str(stage["name"]), candidate, seed),
            candidate=candidate,
            stage=str(stage["name"]),
            seed=seed,
            epochs=float(stage["epochs"]),
            max_steps=int(stage["max_steps"]),
            evaluation_strategy="steps" if int(stage["max_steps"]) > 0 else "epoch",
            eval_steps=int(stage["eval_steps"]),
            parent_run_id=run_id(parent_stage, candidate, seed) if parent_stage else None,
        )
        for candidate in candidates
    ]


def full_run_specs(
    config: Mapping[str, Any],
    finalists: Sequence[str],
) -> list[RunSpec]:
    stage = config["full"]
    # Seed-major ordering gives each finalist an equal chance before the next seed
    # if the hard deadline stops the session.
    return [
        RunSpec(
            run_id=run_id(str(stage["name"]), candidate, int(seed)),
            candidate=candidate,
            stage=str(stage["name"]),
            seed=int(seed),
            epochs=float(stage["epochs"]),
            max_steps=int(stage["max_steps"]),
            evaluation_strategy="steps" if int(stage["max_steps"]) > 0 else "epoch",
            eval_steps=int(stage["eval_steps"]),
        )
        for seed in config["seeds"]
        for candidate in finalists
    ]


def _keep_count(stage: Mapping[str, Any], count: int, minimum: int = 1) -> int:
    if "keep_count" in stage:
        keep = int(stage["keep_count"])
    else:
        keep = math.ceil(count * float(stage.get("keep_fraction", 1.0)))
    return min(count, max(minimum, keep))


def select_survivors(
    validation_metrics: Mapping[str, float | int | None],
    keep: int | float,
    minimum: int = 1,
) -> list[str]:
    """Rank finite validation metrics only, with deterministic name tie-breaking."""
    ranked = sorted(
        (
            (name, float(value))
            for name, value in validation_metrics.items()
            if value is not None and math.isfinite(float(value))
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if isinstance(keep, float):
        if not 0 < keep <= 1:
            raise ValueError("fractional keep must be in (0, 1]")
        count = math.ceil(len(ranked) * keep)
    else:
        count = keep
    count = min(len(ranked), max(minimum, count))
    return [name for name, _ in ranked[:count]]


def build_schedule(
    config: Mapping[str, Any] | None = None,
    rankings: Mapping[str, Sequence[str]] | None = None,
) -> list[RunSpec]:
    """Build the stage shape; supplied rankings replace placeholder config order."""
    config = validate_matrix_config(config or default_config())
    rankings = rankings or {}
    current = [candidate["name"] for candidate in config["candidates"]]
    schedule = runs_for_stage(config, config["smoke"], current)
    smoke_name = str(config["smoke"]["name"])
    current = list(rankings.get(smoke_name, current))
    current = current[: _keep_count(config["smoke"], len(current))]
    parent_stage = smoke_name
    for stage in config["screens"]:
        schedule.extend(runs_for_stage(config, stage, current, parent_stage))
        current = list(rankings.get(str(stage["name"]), current))
        current = current[: _keep_count(stage, len(current), int(config["finalist_count"]))]
        parent_stage = str(stage["name"])
    finalists = current[: int(config["finalist_count"])]
    schedule.extend(full_run_specs(config, finalists))
    return schedule


def reconcile_resume_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Turn crash-left running entries into resumable interrupted entries."""
    recovered = copy.deepcopy(dict(state))
    for row in recovered.get("runs", {}).values():
        if row.get("status") == "running":
            row["status"] = "interrupted"
            row["interrupted_at"] = iso_now()
    recovered["active_run_id"] = None
    return recovered


def pending_runs(
    schedule: Iterable[RunSpec],
    state: Mapping[str, Any],
    retry_failed: bool = False,
) -> list[RunSpec]:
    rows = state.get("runs", {})
    pending: list[RunSpec] = []
    for spec in schedule:
        status = rows.get(spec.run_id, {}).get("status", "pending")
        if status == "complete":
            continue
        if status == "failed" and not retry_failed:
            continue
        if status in RESUMABLE_RUN_STATES or (status == "failed" and retry_failed):
            pending.append(spec)
    return pending


def _seconds(value: datetime | float | int) -> float:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("deadline datetimes must include a timezone")
        return value.timestamp()
    return float(value)


def training_cutoff(
    hard_deadline: datetime | float | int,
    reserve_seconds: float = MINIMUM_RESERVE_SECONDS,
) -> float:
    if reserve_seconds < MINIMUM_RESERVE_SECONDS:
        raise ValueError("at least two hours must be reserved after training")
    return _seconds(hard_deadline) - reserve_seconds


def deadline_allows_training(
    now: datetime | float | int,
    hard_deadline: datetime | float | int,
    reserve_seconds: float = MINIMUM_RESERVE_SECONDS,
) -> bool:
    return _seconds(now) < training_cutoff(hard_deadline, reserve_seconds)


def remaining_training_seconds(
    now: datetime | float | int,
    hard_deadline: datetime | float | int,
    reserve_seconds: float = MINIMUM_RESERVE_SECONDS,
) -> float:
    return max(0.0, training_cutoff(hard_deadline, reserve_seconds) - _seconds(now))


def validate_single_gpu(gpu_selector: str) -> str:
    selector = gpu_selector.strip()
    if not selector or selector in {"-1", "none", "None"}:
        raise ValueError("a CUDA GPU selector is required")
    if "," in selector:
        raise ValueError("exactly one GPU may be visible to the matrix")
    return selector


def single_gpu_environment(
    gpu_selector: str,
    seed: int,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(base or os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": validate_single_gpu(gpu_selector),
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "-1",
            "RANK": "0",
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    return environment


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.pending")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def installed_packages() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "protobuf",
        "psutil",
        "safetensors",
        "sentencepiece",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def optional_command(command: Sequence[str], timeout: float = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def find_latest_checkpoint(path: Path) -> Path | None:
    checkpoints: list[tuple[int, Path]] = []
    if path.is_dir():
        for candidate in path.iterdir():
            match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
            has_model = any(
                (candidate / name).is_file()
                for name in (
                    "model.safetensors",
                    "pytorch_model.bin",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                )
            )
            if (
                candidate.is_dir()
                and match
                and (candidate / "trainer_state.json").is_file()
                and has_model
            ):
                checkpoints.append((int(match.group(1)), candidate))
    return max(checkpoints, default=(0, None), key=lambda item: item[0])[1]


def validation_metric(model_dir: Path, metric_name: str) -> float:
    metrics_path = model_dir / "evaluation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    value = metrics.get(metric_name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{metrics_path} has no finite {metric_name}")
    return float(value)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_manifest(
    config: Mapping[str, Any],
    training_dir: Path,
    output_dir: Path,
    model_root: Path,
    checkpoint_root: Path,
    preparation_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    code_paths = {
        "ml/train.py": ROOT / "ml/train.py",
        "ml/improvement_data.py": ROOT / "ml/improvement_data.py",
        "ml/inference.py": ROOT / "ml/inference.py",
        "ml/heuristic.py": ROOT / "ml/heuristic.py",
        "ml/windowing.py": ROOT / "ml/windowing.py",
        "ml/splits.py": ROOT / "ml/splits.py",
        "ml/cuad_category_domain_mapping.json": ROOT / "ml/cuad_category_domain_mapping.json",
        "ml/CUAD_ATTRIBUTION.md": ROOT / "ml/CUAD_ATTRIBUTION.md",
        "ml/PRETRAINED_ATTRIBUTION.md": ROOT / "ml/PRETRAINED_ATTRIBUTION.md",
        "ml/LLAMA_3_1_NOTICE.txt": ROOT / "ml/LLAMA_3_1_NOTICE.txt",
        "ml/runpod_matrix.py": ROOT / "ml/runpod_matrix.py",
        "scripts/improve-models.py": ROOT / "scripts/improve-models.py",
        "ml/requirements-training-gpu.txt": ROOT / "ml/requirements-training-gpu.txt",
    }
    data_names = (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "labels.json",
        "window_config.json",
        "preparation.json",
    )
    immutable = {
        "schema_version": SCHEMA_VERSION,
        "config": config,
        "code_sha256": {
            name: sha256_file(path) for name, path in code_paths.items() if path.is_file()
        },
        "training_data_sha256": {
            name: sha256_file(training_dir / name)
            for name in data_names
            if (training_dir / name).is_file()
        },
        "preparation": preparation_manifest,
        "python": platform.python_version(),
        "packages": installed_packages(),
    }
    return {
        **immutable,
        "matrix_fingerprint": canonical_sha256(immutable),
        "created_at": iso_now(),
        "selection_data": "validation only",
        "test_data_used_for_selection": False,
        "automatic_promotion": False,
        "determinism": {
            "requested": config["execution"]["deterministic"],
            "seeds": config["seeds"],
            "guaranteed_bitwise_reproducibility": False,
            "limitations": (
                "Seeds and deterministic backend options reduce variation; exact results can "
                "still differ with GPU hardware, drivers, kernels, or package versions."
            ),
        },
        "paths": {
            "output": str(output_dir),
            "models": str(model_root),
            "checkpoints": str(checkpoint_root),
            "state": str(output_dir / "matrix-state.json"),
            "heartbeat": str(output_dir / "heartbeat.json"),
            "commands": str(output_dir / "commands"),
            "telemetry": str(output_dir / "runs"),
        },
        "runtime": {
            "platform": platform.platform(),
            "invocation": sys.argv,
            "gpu_probe": optional_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
    }


def verify_manifest_inputs(manifest: Mapping[str, Any], training_dir: Path) -> None:
    """Fail closed if code or data changes after the matrix fingerprint is frozen."""

    mismatches = []
    for name, expected in manifest.get("code_sha256", {}).items():
        path = ROOT / name
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(name)
    for name, expected in manifest.get("training_data_sha256", {}).items():
        path = training_dir / name
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            mismatches.append(f"training-data/{name}")
    if mismatches:
        raise RuntimeError("matrix inputs changed after launch: " + ", ".join(sorted(mismatches)))


def initialize_state(
    path: Path,
    manifest: Mapping[str, Any],
    hard_deadline: datetime,
    reserve_seconds: float,
) -> dict[str, Any]:
    if not deadline_allows_training(datetime.now(UTC), hard_deadline, reserve_seconds):
        raise ValueError("hard deadline leaves no training time before the two-hour reserve")
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("matrix_fingerprint") != manifest["matrix_fingerprint"]:
            raise ValueError("matrix inputs changed; use a new output directory")
        active = state.get("active_run_id")
        active_row = state.get("runs", {}).get(active, {})
        active_pid = active_row.get("pid")
        if (
            active_row.get("status") == "running"
            and active_row.get("host") == platform.node()
            and isinstance(active_pid, int)
            and _pid_is_running(active_pid)
        ):
            raise RuntimeError(f"recorded training process {active_pid} is still running")
        state = reconcile_resume_state(state)
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "matrix_fingerprint": manifest["matrix_fingerprint"],
            "created_at": iso_now(),
            "status": "running",
            "active_run_id": None,
            "runs": {},
            "stages": {},
            "sessions": [],
            "promoted": False,
            "automatic_promotion": False,
        }
    state["status"] = "running"
    state["updated_at"] = iso_now()
    state["sessions"].append(
        {
            "session_id": uuid.uuid4().hex,
            "started_at": iso_now(),
            "hard_deadline": hard_deadline.isoformat(),
            "training_cutoff": datetime.fromtimestamp(
                training_cutoff(hard_deadline, reserve_seconds), UTC
            ).isoformat(),
            "reserve_seconds": reserve_seconds,
        }
    )
    atomic_json(path, state)
    return state


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


@contextmanager
def matrix_lock(path: Path):
    """Cross-platform advisory lock so two orchestrators cannot share one GPU."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    stream.seek(0)
    if stream.read(1) == "":
        stream.write("0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError("another matrix process holds the one-GPU lock") from error
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class MatrixRunner:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        manifest: Mapping[str, Any],
        output_dir: Path,
        model_root: Path,
        checkpoint_root: Path,
        training_dir: Path,
        hard_deadline: datetime,
        reserve_seconds: float,
        gpu: str,
        python: str,
        retry_failed: bool = False,
    ) -> None:
        self.config = validate_matrix_config(config)
        self.manifest = dict(manifest)
        self.output_dir = output_dir
        self.model_root = model_root
        self.checkpoint_root = checkpoint_root
        self.training_dir = training_dir
        self.hard_deadline = hard_deadline
        self.reserve_seconds = reserve_seconds
        self.gpu = validate_single_gpu(gpu)
        self.python = python
        self.retry_failed = retry_failed
        self.state_path = output_dir / "matrix-state.json"
        self.status_path = output_dir / "training-status.json"
        self.heartbeat_path = output_dir / "heartbeat.json"
        self.report_path = output_dir / "matrix-report.json"
        self.state = initialize_state(
            self.state_path,
            manifest,
            hard_deadline,
            reserve_seconds,
        )

    def _save(self) -> None:
        self.state["updated_at"] = iso_now()
        atomic_json(self.state_path, self.state)
        public_state = "training" if self.state["status"] == "running" else self.state["status"]
        atomic_json(
            self.status_path,
            {
                "state": public_state,
                "active_run_id": self.state.get("active_run_id"),
                "updated_at": self.state["updated_at"],
                "report": str(self.report_path),
                "promoted": False,
                "automatic_promotion": False,
            },
        )
        atomic_json(
            self.heartbeat_path,
            {
                "state": public_state,
                "active_run_id": self.state.get("active_run_id"),
                "updated_at": self.state["updated_at"],
                "state_path": str(self.state_path),
            },
        )

    def _run_paths(self, spec: RunSpec) -> dict[str, Path]:
        run_root = self.output_dir / "runs" / spec.run_id
        return {
            "run_root": run_root,
            "model": self.model_root / spec.run_id,
            "checkpoint": self.checkpoint_root / spec.run_id,
            "log": run_root / "stdout.log",
            "heartbeat": run_root / "training-heartbeat.json",
            "orchestrator_heartbeat": run_root / "orchestrator-heartbeat.json",
            "trainer_telemetry": run_root / "trainer-telemetry.jsonl",
            "gpu_telemetry": run_root / "gpu-telemetry.jsonl",
        }

    def _parent_checkpoint(self, spec: RunSpec) -> Path | None:
        # Resume only an interrupted attempt of this exact run. Hugging Face
        # checkpoints include TrainerState and early-stopping callback state;
        # carrying those across fidelity stages can stop the next stage after
        # its first evaluation and invalidates successive-stage comparisons.
        return find_latest_checkpoint(self._run_paths(spec)["checkpoint"])

    def _command(self, spec: RunSpec, resume: Path | None) -> tuple[list[str], dict[str, str]]:
        candidate = candidate_map(self.config)[spec.candidate]
        paths = self._run_paths(spec)
        stage = self._stage_config(spec.stage)
        command = [
            self.python,
            "-m",
            "ml.train",
            "--model-name",
            str(candidate["model_name"]),
            "--revision",
            str(candidate["revision"]),
            "--base-model-license",
            str(candidate["license"]),
            "--base-model-license-url",
            str(candidate["license_url"]),
            "--base-model-source-url",
            str(candidate["source_url"]),
            "--base-model-card-url",
            str(candidate["model_card_url"]),
            "--base-model-license-source-url",
            str(candidate["license_source_url"]),
            "--base-model-metadata-api-url",
            str(candidate["metadata_api_url"]),
            "--base-model-metadata-checked-at",
            str(candidate["metadata_checked_at"]),
            "--candidate-name",
            spec.candidate,
            "--stage",
            spec.stage,
            "--run-id",
            spec.run_id,
            "--training-dir",
            str(self.training_dir),
            "--model-dir",
            str(paths["model"]),
            "--checkpoint-dir",
            str(paths["checkpoint"]),
            "--heartbeat-file",
            str(paths["heartbeat"]),
            "--telemetry-file",
            str(paths["trainer_telemetry"]),
            "--epochs",
            str(spec.epochs),
            "--max-steps",
            str(spec.max_steps),
            "--evaluation-strategy",
            spec.evaluation_strategy,
            "--eval-steps",
            str(spec.eval_steps),
            "--logging-steps",
            str(min(10, spec.eval_steps)),
            "--early-stopping-patience",
            str(stage["early_stopping_patience"]),
            "--learning-rate",
            str(candidate["learning_rate"]),
            "--batch-size",
            str(candidate["batch_size"]),
            "--gradient-accumulation-steps",
            str(candidate["gradient_accumulation_steps"]),
            "--max-length",
            str(candidate["max_length"]),
            "--positive-weight-cap",
            str(candidate["positive_weight_cap"]),
            "--warmup-ratio",
            str(self.config["optimizer"]["warmup_ratio"]),
            "--weight-decay",
            str(self.config["optimizer"]["weight_decay"]),
            "--precision",
            str(self.config["execution"]["precision"]),
            "--seed",
            str(spec.seed),
            "--require-single-gpu",
        ]
        if self.config["execution"]["deterministic"]:
            command.append("--deterministic")
        if candidate["gradient_checkpointing"]:
            command.append("--gradient-checkpointing")
        command.extend(str(item) for item in candidate.get("extra_args") or [])
        if resume:
            command.extend(["--resume-from-checkpoint", str(resume)])
        return command, single_gpu_environment(self.gpu, spec.seed)

    def _stage_config(self, name: str) -> Mapping[str, Any]:
        stages = [self.config["smoke"], *self.config["screens"], self.config["full"]]
        return next(stage for stage in stages if stage["name"] == name)

    def _completed_metric(self, spec: RunSpec, model_dir: Path) -> float:
        metric = validation_metric(model_dir, self.config["selection_metric"])
        provenance = json.loads(
            (model_dir / "training_provenance.json").read_text(encoding="utf-8")
        )
        expected = candidate_map(self.config)[spec.candidate]
        base = provenance["base_model_provenance"]
        matrix_run = provenance["matrix_run"]
        if (
            base.get("exact_revision") != expected["revision"]
            or base.get("license") != expected["license"]
            or matrix_run.get("run_id") != spec.run_id
            or matrix_run.get("candidate") != spec.candidate
            or matrix_run.get("stage") != spec.stage
        ):
            raise ValueError("artifact provenance differs from this matrix run")
        weights = model_dir / "model.safetensors"
        if not weights.is_file() or provenance.get("weights_sha256") != sha256_file(weights):
            raise ValueError("artifact weights do not match their recorded hash")
        return metric

    def _record_command(
        self,
        spec: RunSpec,
        command: Sequence[str],
        environment: Mapping[str, str],
        paths: Mapping[str, Path],
        resume: Path | None,
    ) -> Path:
        row = self.state["runs"].setdefault(spec.run_id, {})
        attempt = int(row.get("attempts", 0)) + 1
        target = self.output_dir / "commands" / f"{spec.run_id}--attempt-{attempt}.json"
        atomic_json(
            target,
            {
                "schema_version": SCHEMA_VERSION,
                "matrix_fingerprint": self.manifest["matrix_fingerprint"],
                "created_at": iso_now(),
                "cwd": str(ROOT),
                "argv": list(command),
                "environment": {
                    key: environment[key]
                    for key in (
                        "CUBLAS_WORKSPACE_CONFIG",
                        "CUDA_VISIBLE_DEVICES",
                        "HF_HUB_DISABLE_XET",
                        "LOCAL_RANK",
                        "PYTHONHASHSEED",
                        "TOKENIZERS_PARALLELISM",
                        "WORLD_SIZE",
                    )
                },
                "resume_from_checkpoint": str(resume) if resume else None,
                "paths": {key: str(value) for key, value in paths.items()},
                "candidate_provenance": candidate_map(self.config)[spec.candidate],
            },
        )
        return target

    def _terminate(self, process: subprocess.Popen[Any], grace_seconds: float = 30) -> None:
        if process.poll() is not None:
            return
        if grace_seconds <= 0:
            with suppress(ProcessLookupError):
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=grace_seconds)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)

    def _execute_process(
        self,
        spec: RunSpec,
        command: Sequence[str],
        environment: Mapping[str, str],
        paths: Mapping[str, Path],
    ) -> ProcessResult:
        for path in (paths["run_root"], paths["checkpoint"], paths["model"]):
            path.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        cutoff = training_cutoff(self.hard_deadline, self.reserve_seconds)
        if time.time() >= cutoff:
            return ProcessResult(returncode=124, timed_out=True, duration_seconds=0.0)
        with paths["log"].open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(command),
                cwd=ROOT,
                env=dict(environment),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            self.state["runs"][spec.run_id].update({"pid": process.pid, "host": platform.node()})
            self._save()
            timed_out = False
            try:
                while process.poll() is None:
                    if time.time() >= cutoff:
                        timed_out = True
                        self._terminate(process, grace_seconds=0)
                        break
                    heartbeat = {
                        "state": "training",
                        "run_id": spec.run_id,
                        "pid": process.pid,
                        "updated_at": iso_now(),
                        "seconds_until_training_cutoff": max(0.0, cutoff - time.time()),
                        "trainer_telemetry_path": str(paths["trainer_telemetry"]),
                        "gpu_telemetry_path": str(paths["gpu_telemetry"]),
                    }
                    atomic_json(paths["orchestrator_heartbeat"], heartbeat)
                    atomic_json(self.heartbeat_path, heartbeat)
                    gpu = optional_command(
                        [
                            "nvidia-smi",
                            "-i",
                            self.gpu,
                            "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ],
                        timeout=5,
                    )
                    with paths["gpu_telemetry"].open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(
                                {"event": "gpu_sample", "at": iso_now(), "result": gpu},
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    if time.time() >= cutoff:
                        timed_out = True
                        self._terminate(process, grace_seconds=0)
                        break
                    wait_seconds = min(30.0, max(0.1, cutoff - time.time()))
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=wait_seconds)
            except BaseException:
                self._terminate(process)
                raise
            returncode = process.wait()
        return ProcessResult(
            returncode=returncode,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - start,
        )

    def _run_one(self, spec: RunSpec) -> bool:
        verify_manifest_inputs(self.manifest, self.training_dir)
        existing = self.state["runs"].get(spec.run_id, {})
        if existing.get("status") == "complete":
            try:
                metric = self._completed_metric(spec, self._run_paths(spec)["model"])
            except (FileNotFoundError, ValueError):
                existing["status"] = "interrupted"
            else:
                existing["validation_metric"] = metric
                return True
        if existing.get("status") == "failed" and not self.retry_failed:
            return False
        if not deadline_allows_training(
            datetime.now(UTC), self.hard_deadline, self.reserve_seconds
        ):
            self.state["status"] = "deadline_reached"
            self._save()
            return False

        paths = self._run_paths(spec)
        resume = self._parent_checkpoint(spec)
        command, environment = self._command(spec, resume)
        command_path = self._record_command(
            spec,
            command,
            environment,
            paths,
            resume,
        )
        attempts = int(existing.get("attempts", 0)) + 1
        command_record = {
            "attempt": attempts,
            "path": str(command_path),
            "sha256": sha256_file(command_path),
        }
        self.state["runs"][spec.run_id] = {
            **existing,
            **asdict(spec),
            "status": "running",
            "attempts": attempts,
            "started_at": iso_now(),
            "resume_from_checkpoint": str(resume) if resume else None,
            "command_path": str(command_path),
            "command_sha256": command_record["sha256"],
            "command_history": [*existing.get("command_history", []), command_record],
            "paths": {key: str(value) for key, value in paths.items()},
        }
        self.state["active_run_id"] = spec.run_id
        self._save()
        result = self._execute_process(spec, command, environment, paths)
        row = self.state["runs"][spec.run_id]
        row.update(
            {
                "returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "finished_at": iso_now(),
            }
        )
        self.state["active_run_id"] = None
        if result.timed_out:
            row["status"] = "deadline_paused"
            self.state["status"] = "deadline_reached"
            self._save()
            return False
        if result.returncode != 0:
            row["status"] = "failed"
            self._save()
            return False
        try:
            metric = self._completed_metric(spec, paths["model"])
        except (FileNotFoundError, KeyError, ValueError) as error:
            row.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            self._save()
            return False
        row.update({"status": "complete", "validation_metric": metric})
        self._save()
        return True

    def _execute_stage(
        self,
        stage: Mapping[str, Any],
        names: Sequence[str],
        parent_stage: str | None,
    ) -> list[str] | None:
        specs = runs_for_stage(self.config, stage, names, parent_stage)
        for spec in specs:
            self._run_one(spec)
            if self.state["status"] == "deadline_reached":
                return None
        values = {
            spec.candidate: self.state["runs"].get(spec.run_id, {}).get("validation_metric")
            for spec in specs
            if self.state["runs"].get(spec.run_id, {}).get("status") == "complete"
        }
        minimum = (
            int(self.config["finalist_count"])
            if stage["name"] != self.config["smoke"]["name"]
            else 1
        )
        keep = (
            int(stage["keep_count"])
            if "keep_count" in stage
            else float(stage.get("keep_fraction", 1.0))
        )
        survivors = select_survivors(values, keep, minimum)
        ranking = select_survivors(values, len(values), 1)
        self.state["stages"][stage["name"]] = {
            "status": "complete",
            "validation_metric": self.config["selection_metric"],
            "metrics": values,
            "ranking": ranking,
            "survivors": survivors,
            "pruned": sorted(set(names) - set(survivors)),
            "selection_data": "validation",
        }
        self._save()
        return survivors

    def _plan(self) -> int:
        schedule = build_schedule(self.config)
        plan = {
            "schema_version": SCHEMA_VERSION,
            "matrix_fingerprint": self.manifest["matrix_fingerprint"],
            "runs": [asdict(spec) for spec in schedule],
            "sequential": True,
            "max_concurrent_runs": 1,
            "gpus_per_run": 1,
            "automatic_promotion": False,
        }
        atomic_json(self.output_dir / "plan.json", plan)
        self.state["status"] = "planned"
        self.state["planned_runs"] = len(schedule)
        self._save()
        return 0

    def _final_report(self, finalists: Sequence[str]) -> dict[str, Any]:
        full_specs = full_run_specs(self.config, finalists)
        by_candidate: dict[str, list[float]] = {name: [] for name in finalists}
        runs: list[dict[str, Any]] = []
        for spec in full_specs:
            row = self.state["runs"].get(spec.run_id, {})
            runs.append(
                {
                    "run_id": spec.run_id,
                    "candidate": spec.candidate,
                    "seed": spec.seed,
                    "status": row.get("status", "pending"),
                    "validation_metric": row.get("validation_metric"),
                    "paths": row.get("paths", {}),
                }
            )
            if row.get("status") == "complete":
                by_candidate[spec.candidate].append(float(row["validation_metric"]))
        aggregate = []
        expected_seeds = len(self.config["seeds"])
        for candidate, values in by_candidate.items():
            if len(values) > 1:
                validation_stdev = statistics.pstdev(values)
            elif values:
                validation_stdev = 0.0
            else:
                validation_stdev = None
            seeds_complete = len(values) == expected_seeds
            aggregate.append(
                {
                    "candidate": candidate,
                    "completed_seeds": len(values),
                    "expected_seeds": expected_seeds,
                    "seeds_complete": seeds_complete,
                    "validation_mean": (statistics.fmean(values) if seeds_complete else None),
                    "partial_validation_mean": (statistics.fmean(values) if values else None),
                    "validation_stdev": validation_stdev,
                }
            )
        aggregate.sort(
            key=lambda row: (
                not row["seeds_complete"],
                -(row["validation_mean"] if row["validation_mean"] is not None else -math.inf),
                row["candidate"],
            )
        )
        complete = (
            len(finalists) == int(self.config["finalist_count"])
            and len(runs) == len(finalists) * len(self.config["seeds"])
            and all(row["status"] == "complete" for row in runs)
        )
        report_state = "complete" if complete else self.state["status"]
        if report_state == "running":
            report_state = "incomplete"
        return {
            "schema_version": SCHEMA_VERSION,
            "state": report_state,
            "matrix_fingerprint": self.manifest["matrix_fingerprint"],
            "selection_metric": self.config["selection_metric"],
            "selection_data": "validation only",
            "test_data_evaluated": False,
            "screening": self.state["stages"],
            "finalists": list(finalists),
            "full_runs": runs,
            "aggregate_validation": aggregate,
            "promoted": False,
            "automatic_promotion": False,
            "promotion_decision": "not_performed; independent downstream and human review required",
            "manifest_path": str(self.output_dir / "matrix-manifest.json"),
            "state_path": str(self.state_path),
            "completed_at": iso_now() if complete else None,
        }

    def _run(self, dry_run: bool = False) -> int:
        if dry_run:
            return self._plan()
        names = [candidate["name"] for candidate in self.config["candidates"]]
        parent_stage: str | None = None
        stages = [self.config["smoke"], *self.config["screens"]]
        for stage in stages:
            survivors = self._execute_stage(stage, names, parent_stage)
            if survivors is None:
                atomic_json(self.report_path, self._final_report([]))
                return 0
            names = survivors
            parent_stage = str(stage["name"])
        finalist_count = int(self.config["finalist_count"])
        finalists = names[:finalist_count]
        if len(finalists) < finalist_count:
            self.state["status"] = "insufficient_candidates"
            self._save()
            atomic_json(self.report_path, self._final_report(finalists))
            return 1
        for spec in full_run_specs(self.config, finalists):
            self._run_one(spec)
            if self.state["status"] == "deadline_reached":
                atomic_json(self.report_path, self._final_report(finalists))
                return 0
        report = self._final_report(finalists)
        atomic_json(self.report_path, report)
        self.state["status"] = report["state"]
        self._save()
        return 0 if report["state"] == "complete" else 1

    def run(self, dry_run: bool = False) -> int:
        try:
            return self._run(dry_run)
        except BaseException as error:
            active = self.state.get("active_run_id")
            if active and self.state["runs"].get(active, {}).get("status") == "running":
                self.state["runs"][active].update(
                    {
                        "status": "interrupted",
                        "interrupted_at": iso_now(),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            self.state["active_run_id"] = None
            self.state["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            self.state["error"] = f"{type(error).__name__}: {error}"
            self._save()
            raise


def parse_deadline(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include a timezone")
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-data", type=Path, default=Path("ml/data/comparison"))
    parser.add_argument("--training-dir", type=Path, default=Path("ml/data/improvement"))
    parser.add_argument("--raw-data", type=Path, default=Path("ml/data/CUAD_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("artifacts/models/runpod-matrix"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("artifacts/checkpoints/runpod-matrix"),
    )
    parser.add_argument(
        "--deadline",
        help="Hard UTC/offset deadline; TRAINING_HARD_DEADLINE is used when omitted",
    )
    parser.add_argument("--max-runtime-hours", type=float, default=24.0)
    parser.add_argument("--reserve-hours", type=float, default=2.0)
    parser.add_argument("--gpu", default=os.getenv("RUNPOD_MATRIX_GPU", "0"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reserve_seconds = args.reserve_hours * 3600
    if reserve_seconds < MINIMUM_RESERVE_SECONDS:
        raise ValueError("reserve-hours cannot be less than 2")
    deadline_value = args.deadline or os.getenv("TRAINING_HARD_DEADLINE")
    if deadline_value:
        hard_deadline = parse_deadline(deadline_value)
    else:
        hard_deadline = datetime.now(UTC) + timedelta(hours=args.max_runtime_hours)
    if not deadline_allows_training(datetime.now(UTC), hard_deadline, reserve_seconds):
        raise ValueError("hard deadline leaves no training time before the two-hour reserve")

    source_data = _repo_path(args.source_data)
    training_dir = _repo_path(args.training_dir)
    raw_data = _repo_path(args.raw_data)
    output_dir = _repo_path(args.output_dir)
    model_root = _repo_path(args.model_root)
    checkpoint_root = _repo_path(args.checkpoint_root)
    config_path = _repo_path(args.config) if args.config else None
    config = load_matrix_config(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    with matrix_lock(output_dir / "matrix.lock"):
        # Preparation is intentionally before manifest/state creation and before
        # any model command. A changed fingerprint cannot join an old matrix.
        preparation_manifest = prepare(source_data, training_dir, raw_data)
        manifest = build_manifest(
            config,
            training_dir,
            output_dir,
            model_root,
            checkpoint_root,
            preparation_manifest,
        )
        manifest_path = output_dir / "matrix-manifest.json"
        if manifest_path.is_file():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("matrix_fingerprint") != manifest["matrix_fingerprint"]:
                raise ValueError("matrix inputs changed; use a new output directory")
        atomic_json(manifest_path, manifest)
        runner = MatrixRunner(
            config=config,
            manifest=manifest,
            output_dir=output_dir,
            model_root=model_root,
            checkpoint_root=checkpoint_root,
            training_dir=training_dir,
            hard_deadline=hard_deadline,
            reserve_seconds=reserve_seconds,
            gpu=args.gpu,
            python=args.python,
            retry_failed=args.retry_failed,
        )
        return runner.run(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
