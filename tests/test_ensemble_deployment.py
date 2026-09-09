import hashlib
import json
from pathlib import Path

import pytest

from ml.inference import _ensemble_artifact_settings
from ml.package_ensemble import DEFAULT_MEMBERS, MEMBER_FILES, TOKENIZER_FILES, build_ensemble


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backup(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "backup"
    root.mkdir()
    revision = "a" * 40
    model_id = "meta-llama/Llama-3.1-8B"
    members = []
    member_model_ids = []
    member_hashes = []
    for index, name in enumerate(DEFAULT_MEMBERS):
        path = root / name
        path.mkdir()
        weights = path / "adapter_model.safetensors"
        weights.write_bytes(f"adapter-{index}".encode())
        weight_hash = _digest(weights)
        member_id = f"member-{index}"
        _write_json(
            path / "adapter_config.json",
            {"base_model_name_or_path": model_id, "revision": revision},
        )
        _write_json(
            path / "base_model_provenance.json",
            {"model_id": model_id, "exact_revision": revision},
        )
        _write_json(
            path / "label_mapping.json",
            {"label2id": {"termination": 0}, "id2label": {"0": "termination"}},
        )
        _write_json(
            path / "training_provenance.json",
            {
                "artifact_type": "lora_adapter",
                "model_id": member_id,
                "weights_file": weights.name,
                "weights_sha256": weight_hash,
                "tokenizer_max_length": 512,
                "windowing": {"window_chars": 1800, "stride_chars": 900},
                "base_model_provenance": {
                    "model_id": model_id,
                    "exact_revision": revision,
                    "resolved_revision": revision,
                },
            },
        )
        _write_json(
            path / "window_config.json",
            {"window_chars": 1800, "stride_chars": 900},
        )
        for filename in TOKENIZER_FILES:
            (path / filename).write_text(f"{filename}-content", encoding="utf-8")
        files = {
            filename: {
                "bytes": (path / filename).stat().st_size,
                "sha256": _digest(path / filename),
            }
            for filename in (*MEMBER_FILES, *TOKENIZER_FILES)
        }
        members.append(
            {
                "name": name,
                "model_id": member_id,
                "weights_sha256": weight_hash,
                "files": files,
            }
        )
        member_model_ids.append(member_id)
        member_hashes.append(weight_hash)
    _write_json(
        root / "BACKUP_MANIFEST.json",
        {"immutable_copy": True, "created_at": "2026-09-09T00:00:00Z", "members": members},
    )

    base = tmp_path / "base"
    base.mkdir()
    _write_json(base / "config.json", {"model_type": "llama"})
    (base / "model.safetensors").write_bytes(b"base-weights")
    thresholds = tmp_path / "thresholds.json"
    _write_json(
        thresholds,
        {
            "model_id": "llama-ensemble",
            "promotion_authorized": False,
            "member_model_ids": member_model_ids,
            "member_weights_sha256": member_hashes,
            "selection_metrics": {"micro_f1": 0.79},
            "thresholds": {"termination": 0.5},
        },
    )
    return root, base, thresholds


def test_package_ensemble_is_self_contained_and_hash_verified(tmp_path: Path) -> None:
    backup, base, thresholds = _backup(tmp_path)
    output = tmp_path / "ensemble"

    config = build_ensemble(backup, base, thresholds, output)
    settings = _ensemble_artifact_settings(output)

    assert config["model_id"] == settings["model_id"] == "llama-ensemble"
    assert settings["member_names"] == ("seed-17", "seed-29", "seed-43")
    assert len(config["base_model"]["files"]) == 2

    (output / "base_model/model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="verification failed"):
        _ensemble_artifact_settings(output)
