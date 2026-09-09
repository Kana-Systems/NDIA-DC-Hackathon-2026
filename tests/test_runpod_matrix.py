import hashlib
import json
import math
from datetime import UTC, datetime, timedelta

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
    pending_runs,
    reconcile_resume_state,
    select_survivors,
    single_gpu_environment,
    training_cutoff,
    validate_matrix_config,
    validate_single_gpu,
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
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint-21"


def test_deadline_always_reserves_last_two_hours():
    deadline = 10_000.0

    assert training_cutoff(deadline) == deadline - MINIMUM_RESERVE_SECONDS
    assert deadline_allows_training(deadline - MINIMUM_RESERVE_SECONDS - 1, deadline)
    assert not deadline_allows_training(deadline - MINIMUM_RESERVE_SECONDS, deadline)
    with pytest.raises(ValueError, match="two hours"):
        training_cutoff(deadline, MINIMUM_RESERVE_SECONDS - 1)


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


def test_runner_executes_sequentially_resumes_rungs_and_never_promotes(
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
        candidate = next(
            item for item in config["candidates"] if item["name"] == spec.candidate
        )
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
        (paths["checkpoint"] / "checkpoint-1").mkdir(parents=True, exist_ok=True)
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
    assert all("--resume-from-checkpoint" in command for command in screen_commands)
    report = json.loads((output / "matrix-report.json").read_text(encoding="utf-8"))
    assert report["state"] == "complete"
    assert report["automatic_promotion"] is False
    assert report["test_data_evaluated"] is False
