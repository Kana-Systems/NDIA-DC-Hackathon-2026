import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from ml.train import (
    build_parser,
    enable_lora_checkpoint_input_gradients,
    ensure_tokenizer_padding,
    lora_backbone_state_dict,
    lora_is_enabled,
    merge_lora_backbone_state,
    package_artifacts,
    parameter_counts,
    parse_lora_target_modules,
    resolve_torch_dtype,
    save_trained_model,
    single_label_metrics,
    validate_lora_args,
)


def test_lora_flags_default_to_full_finetune():
    args = build_parser().parse_args([])

    assert args.lora_rank == 0
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.05
    assert args.lora_target_modules is None
    assert args.save_adapter_only is False
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
    with pytest.raises(ValueError, match="requires LoRA"):
        validate_lora_args(Namespace(lora_rank=0, save_adapter_only=True))
    with pytest.raises(ValueError, match="requires LoRA"):
        validate_lora_args(
            Namespace(
                lora_rank=0,
                save_adapter_only=False,
                lora_init_adapter="previous-adapter",
            )
        )


def test_lora_transfer_keeps_backbone_tensors_and_discards_task_head():
    state = {
        "base.model.layers.0.q_proj.lora_A.weight": "a",
        "base.model.layers.0.q_proj.lora_B.weight": "b",
        "base.model.score.modules_to_save.default.weight": "head",
    }

    assert lora_backbone_state_dict(state) == {
        "base.model.layers.0.q_proj.lora_A.weight": "a",
        "base.model.layers.0.q_proj.lora_B.weight": "b",
    }
    with pytest.raises(ValueError, match="no transferable"):
        lora_backbone_state_dict({"base.model.score.weight": "head"})


def test_lora_transfer_preserves_new_task_head_required_by_peft_loader():
    current = {
        "base.model.layers.0.q_proj.lora_A.weight": "new-a",
        "base.model.layers.0.q_proj.lora_B.weight": "new-b",
        "base.model.score.weight": "new-task-head",
    }
    source = {
        "base.model.layers.0.q_proj.lora_A.weight": "source-a",
        "base.model.layers.0.q_proj.lora_B.weight": "source-b",
        "base.model.score.weight": "old-task-head",
    }

    assert merge_lora_backbone_state(current, source) == {
        "base.model.layers.0.q_proj.lora_A.weight": "source-a",
        "base.model.layers.0.q_proj.lora_B.weight": "source-b",
        "base.model.score.weight": "new-task-head",
    }


def test_gradient_checkpointed_lora_enables_input_gradients():
    calls = []
    model = SimpleNamespace(enable_input_require_grads=lambda: calls.append("enabled"))

    assert (
        enable_lora_checkpoint_input_gradients(
            model,
            Namespace(lora_rank=128, gradient_checkpointing=True),
        )
        is True
    )
    assert calls == ["enabled"]
    assert (
        enable_lora_checkpoint_input_gradients(
            model,
            Namespace(lora_rank=128, gradient_checkpointing=False),
        )
        is False
    )


def test_adapter_only_save_does_not_merge_base_model(tmp_path):
    calls = []
    model = SimpleNamespace(
        save_pretrained=lambda path, **kwargs: calls.append(("adapter", path, kwargs)),
        merge_and_unload=lambda: calls.append(("merge",)),
    )
    tokenizer = SimpleNamespace(
        save_pretrained=lambda path: calls.append(("tokenizer", path)),
    )
    args = Namespace(model_dir=tmp_path, lora_rank=64, save_adapter_only=True)

    save_trained_model(SimpleNamespace(model=model), tokenizer, args)

    assert calls == [
        ("adapter", str(tmp_path), {"safe_serialization": True}),
        ("tokenizer", str(tmp_path)),
    ]


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


def test_single_label_metrics_include_macro_f1():
    metrics = single_label_metrics(
        [[5, 0, 0], [0, 5, 0], [0, 5, 0], [0, 0, 5]],
        [0, 1, 2, 2],
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["micro_f1"] == 0.75
    assert metrics["macro_precision"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert metrics["macro_recall"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert metrics["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)


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


def test_package_artifacts_hashes_adapter_without_merged_weights(tmp_path):
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
            "--lora-rank",
            "64",
            "--save-adapter-only",
        ]
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "adapter_model.safetensors").write_bytes(b"adapter")
    (model / "adapter_config.json").write_text("{}")

    package_artifacts(
        model,
        ["Termination"],
        args,
        {"eval_micro_f1_tuned": 0.75},
        {"window_chars": 1800, "stride_chars": 900},
    )

    provenance = (model / "training_provenance.json").read_text(encoding="utf-8")
    assert '"artifact_type": "lora_adapter"' in provenance
    assert '"weights_file": "adapter_model.safetensors"' in provenance
    assert '"adapter_model.safetensors":' in provenance


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


def test_external_task_package_uses_separate_attribution_and_model_prefix(tmp_path):
    attribution = tmp_path / "EXTERNAL_ATTRIBUTION.md"
    attribution.write_text("# External dataset\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--model-name",
            "meta-llama/Llama-3.1-8B",
            "--base-model-license",
            "llama3.1",
            "--source-dataset",
            "MAUD v1",
            "--source-license",
            "CC BY 4.0",
            "--dataset-attribution-file",
            str(attribution),
            "--model-id-prefix",
            "Llama-3.1-MAUD",
            "--training-dir",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "model"),
        ]
    )
    (tmp_path / "model").mkdir()

    package_artifacts(
        tmp_path / "model",
        ["0", "1"],
        args,
        {"eval_macro_f1": 0.5},
        {"window_chars": 1800, "stride_chars": 1800},
    )

    provenance = json.loads(
        (tmp_path / "model" / "training_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_dataset"] == "MAUD v1"
    assert provenance["source_license"] == "CC BY 4.0"
    assert provenance["model_id"].startswith("Llama-3.1-MAUD-")
    assert (tmp_path / "model" / "DATASET_ATTRIBUTION.md").is_file()
    assert not (tmp_path / "model" / "CUAD_ATTRIBUTION.md").exists()
    assert not (tmp_path / "model" / "cuad_category_domain_mapping.json").exists()
