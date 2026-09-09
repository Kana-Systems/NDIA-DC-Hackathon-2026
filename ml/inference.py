"""SageMaker-compatible windowed inference with a deterministic fallback."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ml.heuristic import HeuristicClassifier
    from ml.windowing import (
        DEFAULT_WINDOW_CHARS,
        DEFAULT_WINDOW_STRIDE,
        sliding_text_windows,
    )
except ImportError:  # SageMaker places source_dir files at the code archive root.
    from heuristic import HeuristicClassifier
    from windowing import DEFAULT_WINDOW_CHARS, DEFAULT_WINDOW_STRIDE, sliding_text_windows


@dataclass
class TransformerClassifier:
    predictor: Any
    window_chars: int = DEFAULT_WINDOW_CHARS
    stride_chars: int = DEFAULT_WINDOW_STRIDE
    tokenizer_max_length: int = 512
    model_id: str = "fine-tuned-transformer"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_artifact_settings(path: Path) -> dict[str, Any]:
    adapter_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    provenance_path = path / "training_provenance.json"
    mapping_path = path / "label_mapping.json"
    required = (adapter_path, weights_path, provenance_path, mapping_path)
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise RuntimeError(f"incomplete LoRA adapter artifact: missing {', '.join(missing)}")

    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    base = provenance.get("base_model_provenance", {})
    base_model = base.get("model_id")
    revision = base.get("exact_revision")
    if not isinstance(base_model, str) or not base_model.strip():
        raise RuntimeError("LoRA adapter provenance requires a base model ID")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise RuntimeError("LoRA adapter provenance requires an immutable base revision")
    if base.get("resolved_revision") != revision:
        raise RuntimeError("LoRA adapter base revision does not match its resolved revision")
    if adapter.get("base_model_name_or_path") != base_model:
        raise RuntimeError("LoRA adapter config and provenance name different base models")
    adapter_revision = adapter.get("revision")
    if adapter_revision not in (None, revision):
        raise RuntimeError("LoRA adapter config and provenance name different revisions")
    if provenance.get("artifact_type") != "lora_adapter":
        raise RuntimeError("LoRA adapter provenance has the wrong artifact type")
    if provenance.get("weights_file") != weights_path.name:
        raise RuntimeError("LoRA adapter provenance has the wrong weights filename")
    if provenance.get("weights_sha256") != _sha256(weights_path):
        raise RuntimeError("LoRA adapter weights do not match their recorded hash")

    label_to_id = mapping.get("label2id")
    id_to_label = mapping.get("id2label")
    if not isinstance(label_to_id, dict) or not label_to_id:
        raise RuntimeError("LoRA adapter requires a non-empty label mapping")
    expected_id_to_label = {str(index): label for label, index in label_to_id.items()}
    if id_to_label != expected_id_to_label:
        raise RuntimeError("LoRA adapter label mappings are inconsistent")
    return {
        "base_model": base_model,
        "revision": revision,
        "label2id": label_to_id,
        "id2label": {int(index): label for index, label in id_to_label.items()},
    }


def _artifact_settings(path: Path) -> dict[str, Any]:
    model_config_path = path / "config.json"
    provenance_path = path / "training_provenance.json"
    window_path = path / "window_config.json"
    model_config = (
        json.loads(model_config_path.read_text(encoding="utf-8"))
        if model_config_path.exists()
        else {}
    )
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    )
    windowing = provenance.get("windowing", {})
    if window_path.exists():
        windowing = json.loads(window_path.read_text(encoding="utf-8"))
    return {
        "window_chars": int(
            windowing.get(
                "window_chars",
                model_config.get("contract_window_chars", DEFAULT_WINDOW_CHARS),
            )
        ),
        "stride_chars": int(
            windowing.get(
                "stride_chars",
                model_config.get("contract_window_stride_chars", DEFAULT_WINDOW_STRIDE),
            )
        ),
        "tokenizer_max_length": int(
            provenance.get(
                "tokenizer_max_length",
                model_config.get("contract_tokenizer_max_length", 512),
            )
        ),
        "model_id": str(provenance.get("model_id", "fine-tuned-transformer")),
    }


def model_fn(model_dir: str) -> Any:
    path = Path(model_dir)
    if (path / "linear_config.json").exists():
        from ml.linear import LinearClassifier

        return LinearClassifier(path)
    has_adapter_config = (path / "adapter_config.json").is_file()
    has_adapter_weights = (path / "adapter_model.safetensors").is_file()
    if has_adapter_config != has_adapter_weights:
        raise RuntimeError("incomplete LoRA adapter artifact")
    is_adapter = has_adapter_config and has_adapter_weights
    if not (path / "config.json").exists() and not is_adapter:
        return HeuristicClassifier()
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as error:
        raise RuntimeError(
            "a packaged transformer model requires transformers and torch"
        ) from error
    batch_size = int(os.getenv("MODEL_INFERENCE_BATCH_SIZE", "16"))
    if batch_size < 1:
        raise ValueError("MODEL_INFERENCE_BATCH_SIZE must be positive")
    tokenizer = AutoTokenizer.from_pretrained(path)
    adapter_settings = _adapter_artifact_settings(path) if is_adapter else None
    if adapter_settings is not None:
        model_config = AutoConfig.from_pretrained(
            adapter_settings["base_model"],
            revision=adapter_settings["revision"],
        )
        model_config.num_labels = len(adapter_settings["label2id"])
        model_config.label2id = adapter_settings["label2id"]
        model_config.id2label = adapter_settings["id2label"]
        model_config.problem_type = "multi_label_classification"
        model_config.pad_token_id = tokenizer.pad_token_id
    else:
        model_config = AutoConfig.from_pretrained(path)
    if hasattr(model_config, "reference_compile"):
        model_config.reference_compile = False
    if adapter_settings is not None:
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("a packaged LoRA adapter requires peft") from error
        dtype = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            adapter_settings["base_model"],
            revision=adapter_settings["revision"],
            config=model_config,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, config=model_config)
    predictor = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        batch_size=batch_size,
        device=0 if torch.cuda.is_available() else -1,
    )
    return TransformerClassifier(predictor=predictor, **_artifact_settings(path))


def input_fn(body: str | bytes, content_type: str) -> dict[str, Any]:
    if content_type.split(";", 1)[0].strip() != "application/json":
        raise ValueError("content type must be application/json")
    payload = json.loads(body)
    if "text" in payload:
        payload["texts"] = [payload.pop("text")]
    texts = payload.get("texts")
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(item, str) and item.strip() for item in texts)
    ):
        raise ValueError("request must contain non-empty string 'text' or string array 'texts'")
    payload["threshold"] = float(payload.get("threshold", 0.5))
    return payload


def _windowed_transformer_predictions(
    texts: list[str],
    model: TransformerClassifier,
    threshold: float,
) -> list[dict[str, Any]]:
    windows = []
    owners = []
    for owner, text in enumerate(texts):
        text_windows = sliding_text_windows(text, model.window_chars, model.stride_chars)
        windows.extend(text_windows)
        owners.extend([owner] * len(text_windows))
    raw_results = model.predictor(
        windows,
        truncation=True,
        max_length=model.tokenizer_max_length,
    )
    if raw_results and isinstance(raw_results[0], dict):
        raw_results = [raw_results]
    aggregate: list[dict[str, float]] = [{} for _ in texts]
    for owner, result in zip(owners, raw_results, strict=True):
        for item in result:
            label = str(item["label"])
            score = float(item["score"])
            aggregate[owner][label] = max(aggregate[owner].get(label, 0.0), score)
    return [
        {
            "model_id": model.model_id,
            "labels": [
                {"label": label, "score": round(score, 6)}
                for label, score in sorted(scores.items())
                if score >= threshold
            ],
        }
        for scores in aggregate
    ]


def predict_fn(payload: dict[str, Any], model: Any) -> dict[str, Any]:
    threshold = payload["threshold"]
    if isinstance(model, HeuristicClassifier):
        predictions = model.predict_many(payload["texts"], threshold)
    elif isinstance(model, TransformerClassifier):
        predictions = _windowed_transformer_predictions(
            payload["texts"],
            model,
            threshold,
        )
    else:
        from ml.linear import LinearClassifier

        if not isinstance(model, LinearClassifier):
            raise TypeError("model_fn must return a supported classifier")
        predictions = [model.predict(text, threshold) for text in payload["texts"]]
    return {"predictions": predictions}


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    if accept.split(";", 1)[0].strip() != "application/json":
        raise ValueError("accept must be application/json")
    return json.dumps(prediction, sort_keys=True), "application/json"
