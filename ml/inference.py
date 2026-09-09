"""SageMaker-compatible windowed inference with a deterministic fallback."""

from __future__ import annotations

import json
import os
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
    if not (path / "config.json").exists():
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
    model_config = AutoConfig.from_pretrained(path)
    if hasattr(model_config, "reference_compile"):
        model_config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(path, config=model_config)
    tokenizer = AutoTokenizer.from_pretrained(path)
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
