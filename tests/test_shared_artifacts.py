"""LFS bootstrap checks must fail clearly before dependency installation."""

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def verifier(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/verify-shared-artifacts.py"
    spec = importlib.util.spec_from_file_location("shared_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    return module


def test_missing_download_has_recovery_command(verifier):
    with pytest.raises(SystemExit, match="git lfs pull"):
        verifier.main()


def test_pointer_is_not_treated_as_database(verifier):
    corpus = verifier.ROOT / "artifacts/knowledge/federal-v2.sqlite"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    with pytest.raises(SystemExit, match="LFS pointer, not data"):
        verifier.main()


def test_complete_snapshot_passes_without_ml_dependencies(verifier, capsys):
    model = verifier.ROOT / "artifacts/models/legal-bert-cuad"
    model.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "label_mapping.json",
        "window_config.json",
        "evaluation_metrics.json",
        "cuad_category_domain_mapping.json",
        "CUAD_ATTRIBUTION.md",
        "PRETRAINED_ATTRIBUTION.md",
    ):
        (model / name).write_text("fixture")
    (model / "training_provenance.json").write_text(json.dumps({"model_id": "fixture"}))
    (model.parent / "selected.json").write_text(
        json.dumps(
            {
                "model_id": "fixture",
                "model_path": "artifacts/models/legal-bert-cuad",
            }
        )
    )
    corpus = verifier.ROOT / "artifacts/knowledge/federal-v2.sqlite"
    corpus.parent.mkdir(parents=True)
    with sqlite3.connect(corpus) as db:
        db.execute("CREATE TABLE chunks (id INTEGER)")
        db.executemany("INSERT INTO chunks VALUES (?)", ((i,) for i in range(12574)))
    verifier.main()
    assert "12574 regulatory chunks" in capsys.readouterr().out
