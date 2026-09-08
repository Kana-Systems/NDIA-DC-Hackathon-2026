import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.config import Settings
from app.main import create_app


def _test_client() -> TestClient:
    app = create_app(
        Settings(
            workspace_username="reviewer",
            workspace_password=SecretStr("test-password"),
            bedrock_enabled=False,
        )
    )
    return TestClient(app)


def test_runtime_secrets_have_no_insecure_defaults(monkeypatch) -> None:
    monkeypatch.delenv("WORKSPACE_PASSWORD", raising=False)
    monkeypatch.delenv("GRADIO_PASSWORD", raising=False)
    monkeypatch.delenv("DEMO_JWT_SECRET")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_health_is_public() -> None:
    assert _test_client().get("/health").json() == {"status": "ok"}


def test_openapi_marks_new_acquisition_metadata_required() -> None:
    schema = _test_client().get("/openapi.json").json()
    required = schema["components"]["schemas"]["AcquisitionMetadata"]["required"]

    assert "place_of_performance" in required
    assert "acquisition_stage" in required


def test_review_api_rejects_missing_and_wrong_credentials() -> None:
    client = _test_client()
    missing = client.post("/api/v1/reviews/sample")
    wrong = client.post(
        "/api/v1/reviews/sample",
        auth=("reviewer", "wrong-password"),
    )

    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Basic"
    assert wrong.status_code == 401


def test_review_api_accepts_judge_credentials_without_aws() -> None:
    response = _test_client().post(
        "/api/v1/reviews/sample",
        auth=("reviewer", "test-password"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["synthesis_mode"] == "offline-deterministic"
    assert payload["metadata"]["place_of_performance"] == "Arlington, Virginia"
    assert payload["metadata"]["acquisition_stage"] == "solicitation"
    assert payload["classifier_model_ids"]
    assert payload["ablation_summary"]["retrieved_evidence_count"] >= 1


def test_review_parse_and_analysis_run_off_event_loop(monkeypatch) -> None:
    import app.api as api_module

    original = api_module._parse_and_review
    observed = {"off_loop": False}

    def wrapped(*args):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            observed["off_loop"] = True
        return original(*args)

    monkeypatch.setattr(api_module, "_parse_and_review", wrapped)
    response = _test_client().post(
        "/api/v1/reviews/sample",
        auth=("reviewer", "test-password"),
    )

    assert response.status_code == 200
    assert observed["off_loop"] is True


def test_workspace_credentials_are_configured_and_secret_is_masked() -> None:
    settings = Settings(
        workspace_username="alice",
        workspace_password=SecretStr("super-secret"),
        max_upload_mb=3,
    )

    assert settings.workspace_username == "alice"
    assert settings.max_upload_bytes == 3 * 1024 * 1024
    assert settings.bedrock_model_id == "openai.gpt-5.6-terra"
    assert settings.classifier_enabled is True
    assert "super-secret" not in repr(settings)


def test_retired_ui_route_is_not_served() -> None:
    app = create_app(
        Settings(
            workspace_username="reviewer",
            workspace_password=SecretStr("test-password"),
        )
    )
    response = TestClient(app).get("/ui/")

    assert response.status_code == 404
