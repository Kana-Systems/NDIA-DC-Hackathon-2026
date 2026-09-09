from argparse import Namespace
from types import SimpleNamespace

import pytest

from ml.train import (
    build_parser,
    ensure_tokenizer_padding,
    lora_is_enabled,
    package_artifacts,
    parameter_counts,
    parse_lora_target_modules,
    resolve_torch_dtype,
    validate_lora_args,
)


def test_lora_flags_default_to_full_finetune():
    args = build_parser().parse_args([])

    assert args.lora_rank == 0
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.05
    assert args.lora_target_modules is None
    assert lora_is_enabled(args) is False
    validate_lora_args(args)


def test_lora_flags_and_target_module_parsing():
    args = build_parser().parse_args(
        [
            "--lora-rank",
            "16",
            "--lora-alpha",
            "32",
            "--lora-dropout",
            "0.1",
            "--lora-target-modules",
            "q_proj, v_proj",
        ]
    )

    assert lora_is_enabled(args) is True
    validate_lora_args(args)
    assert parse_lora_target_modules(args.lora_target_modules) == ["q_proj", "v_proj"]
    assert parse_lora_target_modules(None) is None
    with pytest.raises(ValueError, match="lora-alpha"):
        validate_lora_args(Namespace(lora_rank=8, lora_alpha=0, lora_dropout=0.05))
    with pytest.raises(ValueError, match="lora-dropout"):
        validate_lora_args(Namespace(lora_rank=8, lora_alpha=16, lora_dropout=1))


def test_decoder_padding_uses_eos_and_left_side():
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        eos_token_id=2,
        pad_token_id=None,
        padding_side="right",
    )
    config = SimpleNamespace(model_type="mistral")

    metadata = ensure_tokenizer_padding(tokenizer, config)

    assert tokenizer.pad_token == "<eos>"
    assert tokenizer.pad_token_id == 2
    assert tokenizer.padding_side == "left"
    assert config.pad_token_id == 2
    assert metadata["created_pad_token"] is True
    assert metadata["model_type"] == "mistral"


def test_encoder_padding_is_left_unchanged():
    tokenizer = SimpleNamespace(pad_token="[PAD]", pad_token_id=0, padding_side="right")
    config = SimpleNamespace(model_type="bert")

    metadata = ensure_tokenizer_padding(tokenizer, config)

    assert tokenizer.padding_side == "right"
    assert config.pad_token_id == 0
    assert metadata["created_pad_token"] is False


def test_parameter_counts_and_torch_dtype_resolution():
    trainable = SimpleNamespace(numel=lambda: 4, requires_grad=True)
    frozen = SimpleNamespace(numel=lambda: 6, requires_grad=False)
    model = SimpleNamespace(parameters=lambda: [trainable, frozen])
    torch_module = SimpleNamespace(bfloat16="bf16", float16="fp16")

    assert parameter_counts(model) == {
        "trainable": 4,
        "total": 10,
        "trainable_ratio": 0.4,
    }
    assert resolve_torch_dtype(torch_module, "bf16") == "bf16"
    assert resolve_torch_dtype(torch_module, "fp16") == "fp16"
    assert resolve_torch_dtype(torch_module, "fp32") is None


def test_package_artifacts_records_lora_provenance(tmp_path):
    args = build_parser().parse_args(
        [
            "--model-name",
            "example/non-chinese-decoder",
            "--training-dir",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "model"),
        ]
    )
    args.lora_metadata = {
        "enabled": True,
        "rank": 16,
        "parameter_counts": {"trainable": 10, "total": 100, "trainable_ratio": 0.1},
    }
    (tmp_path / "model").mkdir()
    package_artifacts(
        tmp_path / "model",
        ["Termination"],
        args,
        {"eval_micro_f1": 0.1},
        {"window_chars": 1800, "stride_chars": 900},
    )
    provenance = (tmp_path / "model" / "training_provenance.json").read_text(encoding="utf-8")
    assert '"rank": 16' in provenance
    assert '"trainable": 10' in provenance


def test_llama_derivative_packages_notice_and_compliant_model_name(tmp_path):
    args = build_parser().parse_args(
        [
            "--model-name",
            "meta-llama/Llama-3.1-8B",
            "--base-model-license",
            "llama3.1",
            "--training-dir",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "model"),
        ]
    )
    (tmp_path / "model").mkdir()

    package_artifacts(
        tmp_path / "model",
        ["Termination"],
        args,
        {"eval_micro_f1_tuned": 0.1},
        {"window_chars": 1800, "stride_chars": 900},
    )

    notice = (tmp_path / "model" / "NOTICE").read_text(encoding="utf-8")
    provenance = (tmp_path / "model" / "training_provenance.json").read_text(encoding="utf-8")
    assert "Llama 3.1 is licensed" in notice
    assert '"model_id": "Llama-3.1-CUAD-' in provenance
    assert '"NOTICE":' in provenance
