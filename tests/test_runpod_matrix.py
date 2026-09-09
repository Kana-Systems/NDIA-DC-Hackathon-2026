import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ml.runpod_matrix import (
    DEFAULT_SEEDS,
    MINIMUM_RESERVE_SECONDS,
    MatrixRunner,
    ProcessResult,
    build_schedule,
    deadline_allows_training,
    default_config,
    find_latest_checkpoint,
    load_matrix_config,
    pending_runs,
    reconcile_resume_state,
    select_survivors,
    single_gpu_environment,
    training_cutoff,
    validate_matrix_config,
    validate_single_gpu,
    verify_manifest_inputs,
)


def test_default_schedule_halves_then_runs_two_finalists_across_three_seeds():
    config = default_config()
    names = [candidate["name"] for candidate in config["candidates"]]
    rankings = {
        "smoke": names,
        "screen-1": [
            "roberta-large",
            "legal-bert-weighted",
            "deberta-v3-large",
            "caselaw-modernbert-large",
        ],
        "screen-2": ["legal-bert-weighted", "roberta-large"],
    }

    schedule = build_schedule(config, rankings)

    assert [sum(run.stage == stage for run in schedule) for stage in ("smoke", "screen-1")] == [
        4,
        4,
    ]
    assert sum(run.stage == "screen-2" for run in schedule) == 2
    full = [run for run in schedule if run.stage == "full"]
    assert [(run.seed, run.candidate) for run in full] == [
        (seed, candidate)
        for seed in DEFAULT_SEEDS
        for candidate in ("legal-bert-weighted", "roberta-large")
    ]
    assert len(full) == 6
    assert all(run.gpus == 1 for run in schedule)


def test_candidates_capture_pinned_revisions_and_license_provenance():
    candidates = default_config()["candidates"]

    assert [candidate["model_name"] for candidate in candidates] == [
        "nlpaueb/legal-bert-base-uncased",
        "microsoft/deberta-v3-large",
        "ai-law-society-lab/CaseLawModernBERT-large",
        "FacebookAI/roberta-large",
    ]
    assert all(len(candidate["revision"]) == 40 for candidate in candidates)
    assert all(candidate["revision"] in candidate["source_url"] for candidate in candidates)
    assert all(candidate["license"] and candidate["license_source_url"] for candidate in candidates)
    assert candidates[0]["positive_weight_cap"] == 4


def test_xlarge_matrix_is_pinned_and_b300_sized():
    config = load_matrix_config(
        Path(__file__).resolve().parents[1] / "ml" / "runpod_deberta_v2_xlarge.json"
    )

    assert config["finalist_count"] == 2
    assert config["seeds"] == [17, 29, 43]
    candidate = config["candidates"][0]
    assert candidate["model_name"] == "microsoft/deberta-v2-xlarge"
    assert candidate["revision"] == "1d134961d4db8e7e8eb1bc1ab81cb370244c57f7"
    assert candidate["license"] == "mit"
    assert candidate["batch_size"] == 64
    comparator = config["candidates"][1]
    assert comparator["model_name"] == "ai-law-society-lab/CaseLawModernBERT-large"
    assert comparator["batch_size"] == 128
    assert config["full"]["epochs"] == 2.0


def test_caselaw_f1_sweep_compares_weight_caps_and_selects_by_tuned_f1():
    config = load_matrix_config(
        Path(__file__).resolve().parents[1] / "ml" / "runpod_caselaw_f1_sweep.json"
    )

    assert config["selection_metric"] == "eval_micro_f1_tuned"
    assert config["finalist_count"] == 1
    assert config["seeds"] == [17, 29, 43]
    assert len(config["candidates"]) == 2
    assert {candidate["model_name"] for candidate in config["candidates"]} == {
        "ai-law-society-lab/CaseLawModernBERT-large"
    }
    assert {candidate["positive_weight_cap"] for candidate in config["candidates"]} == {
        2.0,
        4.0,
    }
    assert all(
        candidate["extra_args"] == ["--metric-for-best-model", "micro_f1_tuned"]
        for candidate in config["candidates"]
    )
    assert config["smoke"]["keep_count"] == 2
    assert config["screens"][0]["keep_count"] == 1


def test_deberta_f1_sweep_is_tuned_metric_ready_and_b300_sized():
    config = load_matrix_config(
        Path(__file__).resolve().parents[1] / "ml" / "runpod_deberta_f1_sweep.json"
    )

    assert config["selection_metric"] == "eval_micro_f1_tuned"
    assert config["finalist_count"] == 1
    assert config["seeds"] == [17, 29]
    assert len(config["candidates"]) == 3
    assert {candidate["positive_weight_cap"] for candidate in config["candidates"]} == {
        1.0,
        2.0,
        4.0,
    }
    assert all(candidate["batch_size"] == 64 for candidate in config["candidates"])
    assert all(
        candidate["extra_args"] == ["--metric-for-best-model", "micro_f1_tuned"]
        for candidate in config["candidates"]
    )


def test_llama31_matrix_is_pinned_lora_and_uses_compliant_names():
    config = load_matrix_config(
        Path(__file__).resolve().parents[1] / "ml" / "runpod_llama31_8b_f1.json"
    )

    assert config["selection_metric"] == "eval_micro_f1_tuned"
    assert config["seeds"] == [17]
    assert len(config["candidates"]) == 2
    assert all(candidate["name"].startswith("Llama") for candidate in config["candidates"])
    assert all(
        candidate["model_name"] == "meta-llama/Llama-3.1-8B" for candidate in config["candidates"]
    )
    assert all(
        candidate["revision"] == "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
        for candidate in config["candidates"]
    )
    assert all(candidate["license"] == "llama3.1" for candidate in config["candidates"])
    assert {
        candidate["extra_args"][candidate["extra_args"].index("--lora-rank") + 1]
        for candidate in config["candidates"]
    } == {"16", "64"}


def test_validation_ranking_prunes_failures_and_breaks_ties_by_name():
    metrics = {
        "c": 0.7,
        "b": None,
        "a": 0.7,
        "bad": math.nan,
        "d": 0.5,
    }

    assert select_survivors(metrics, 0.5) == ["a", "c"]
    assert select_survivors(metrics, 2) == ["a", "c"]


def test_resume_skips_completed_and_failed_but_recovers_interrupted():
    schedule = build_schedule(default_config())[:3]
    state = {
        "active_run_id": schedule[1].run_id,
        "runs": {
            schedule[0].run_id: {"status": "complete"},
            schedule[1].run_id: {"status": "running"},
            schedule[2].run_id: {"status": "failed"},
        },
    }

    recovered = reconcile_resume_state(state)

    assert recovered["active_run_id"] is None
    assert recovered["runs"][schedule[1].run_id]["status"] == "interrupted"
    assert pending_runs(schedule, recovered) == [schedule[1]]
    assert pending_runs(schedule, recovered, retry_failed=True) == schedule[1:]
    assert state["runs"][schedule[1].run_id]["status"] == "running"


def test_latest_checkpoint_supports_resuming_a_partial_run(tmp_path):
    for step in (8, 21, 13):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"weights")
    incomplete = tmp_path / "checkpoint-34"
    incomplete.mkdir()
    (incomplete / "model.safetensors").write_bytes(b"partial")
    adapter = tmp_path / "checkpoint-40"
    adapter.mkdir()
    (adapter / "trainer_state.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "checkpoint-invalid").mkdir()

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-40"


def test_manifest_input_verification_detects_training_data_mutation(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    split = training / "train.jsonl"
    split.write_text('{"text":"original"}\n')
    manifest = {
        "code_sha256": {},
        "training_data_sha256": {"train.jsonl": hashlib.sha256(split.read_bytes()).hexdigest()},
    }

    verify_manifest_inputs(manifest, training)
    split.write_text('{"text":"changed"}\n')

    with pytest.raises(RuntimeError, match="training-data/train.jsonl"):
        verify_manifest_inputs(manifest, training)


def test_runner_resumes_only_the_same_run_not_a_parent_stage(tmp_path):
    config = default_config()
    spec = next(item for item in build_schedule(config) if item.parent_run_id)
    runner = MatrixRunner(
        config=config,
        manifest={"matrix_fingerprint": "fingerprint"},
        output_dir=tmp_path / "output",
        model_root=tmp_path / "models",
        checkpoint_root=tmp_path / "checkpoints",
        training_dir=tmp_path / "data",
        hard_deadline=datetime.now(UTC) + timedelta(hours=3),
        reserve_seconds=MINIMUM_RESERVE_SECONDS,
        gpu="0",
        python="python",
    )
    parent = tmp_path / "checkpoints" / spec.parent_run_id / "checkpoint-12"
    parent.mkdir(parents=True)
    (parent / "trainer_state.json").write_text("{}")
    (parent / "model.safetensors").write_bytes(b"parent")

    assert runner._parent_checkpoint(spec) is None

    own = tmp_path / "checkpoints" / spec.run_id / "checkpoint-18"
    own.mkdir(parents=True)
    (own / "trainer_state.json").write_text("{}")
    (own / "model.safetensors").write_bytes(b"own")

    assert runner._parent_checkpoint(spec) == own


def test_deadline_always_reserves_last_two_hours():
    deadline = 10_000.0

    assert training_cutoff(deadline) == deadline - MINIMUM_RESERVE_SECONDS
    assert deadline_allows_training(deadline - MINIMUM_RESERVE_SECONDS - 1, deadline)
    assert not deadline_allows_training(deadline - MINIMUM_RESERVE_SECONDS, deadline)
    with pytest.raises(ValueError, match="two hours"):
        training_cutoff(deadline, MINIMUM_RESERVE_SECONDS - 1)


def test_extra_args_are_forwarded_to_the_train_command(tmp_path):
    config = default_config()
    config["candidates"][0]["extra_args"] = ["--lora-rank", "16"]
    spec = build_schedule(config)[0]
    runner = MatrixRunner(
        config=config,
        manifest={"matrix_fingerprint": "fingerprint"},
        output_dir=tmp_path / "output",
        model_root=tmp_path / "models",
        checkpoint_root=tmp_path / "checkpoints",
        training_dir=tmp_path / "data",
        hard_deadline=datetime.now(UTC) + timedelta(hours=3),
        reserve_seconds=MINIMUM_RESERVE_SECONDS,
        gpu="0",
        python="python",
    )

    command, _environment = runner._command(spec, None)

    assert command[-2:] == ["--lora-rank", "16"]


def test_single_gpu_invariant_is_enforced_in_config_and_child_environment():
    environment = single_gpu_environment("GPU-deadbeef", 29, {"KEEP": "yes"})

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-deadbeef"
    assert environment["WORLD_SIZE"] == "1"
    assert environment["LOCAL_RANK"] == "-1"
    assert environment["PYTHONHASHSEED"] == "29"
    assert environment["KEEP"] == "yes"
    with pytest.raises(ValueError, match="exactly one GPU"):
        validate_single_gpu("0,1")

    config = default_config()
    config["execution"]["max_concurrent_runs"] = 2
    with pytest.raises(ValueError, match="exactly one concurrent"):
        validate_matrix_config(config)


def test_runner_executes_sequentially_without_cross_stage_resume_or_promotion(
    tmp_path,
    monkeypatch,
):
    config = default_config()
    config["candidates"] = config["candidates"][:2]
    calls = []

    def fake_execute(self, spec, command, environment, paths):
        calls.append((spec, command, environment))
        model_dir = paths["model"]
        model_dir.mkdir(parents=True, exist_ok=True)
        weights = f"{spec.run_id}-weights".encode()
        (model_dir / "model.safetensors").write_bytes(weights)
        metric = 0.8 if spec.candidate == "legal-bert-weighted" else 0.7
        (model_dir / "evaluation_metrics.json").write_text(
            json.dumps({"eval_micro_f1": metric}),
            encoding="utf-8",
        )
        candidate = next(item for item in config["candidates"] if item["name"] == spec.candidate)
        (model_dir / "training_provenance.json").write_text(
            json.dumps(
                {
                    "base_model_provenance": {
                        "exact_revision": candidate["revision"],
                        "license": candidate["license"],
                    },
                    "matrix_run": {
                        "run_id": spec.run_id,
                        "candidate": spec.candidate,
                        "stage": spec.stage,
                    },
                    "weights_sha256": hashlib.sha256(weights).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        checkpoint = paths["checkpoint"] / "checkpoint-1"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "trainer_state.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(weights)
        return ProcessResult(returncode=0, timed_out=False, duration_seconds=1)

    monkeypatch.setattr(MatrixRunner, "_execute_process", fake_execute)
    output = tmp_path / "output"
    runner = MatrixRunner(
        config=config,
        manifest={"matrix_fingerprint": "fingerprint"},
        output_dir=output,
        model_root=tmp_path / "models",
        checkpoint_root=tmp_path / "checkpoints",
        training_dir=tmp_path / "data",
        hard_deadline=datetime.now(UTC) + timedelta(hours=3),
        reserve_seconds=MINIMUM_RESERVE_SECONDS,
        gpu="0",
        python="python",
    )

    assert runner.run() == 0
    assert runner.state["status"] == "complete"
    assert runner.state["promoted"] is False
    assert len(calls) == 12
    assert all(call[2]["CUDA_VISIBLE_DEVICES"] == "0" for call in calls)
    assert all("--require-single-gpu" in call[1] for call in calls)
    screen_commands = [command for spec, command, _ in calls if spec.stage.startswith("screen")]
    assert all("--resume-from-checkpoint" not in command for command in screen_commands)
    report = json.loads((output / "matrix-report.json").read_text(encoding="utf-8"))
    assert report["state"] == "complete"
    assert report["automatic_promotion"] is False
    assert report["test_data_evaluated"] is False
    assert all(row["seeds_complete"] for row in report["aggregate_validation"])
    assert all(row["validation_mean"] is not None for row in report["aggregate_validation"])
