import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.sample import sample_contract_bytes, sample_metadata


@pytest.fixture
def client():
    return TestClient(create_app(Settings(bedrock_enabled=False)))


def headers(client):
    result = client.post("/api/auth/login", json={"password": "pytest-only-password"})
    assert result.status_code == 200
    return {"Authorization": f"Bearer {result.json()['access_token']}"}


def test_lens_auth_and_native_report(client):
    assert client.get("/api/sources").status_code == 401
    assert client.post("/api/auth/login", json={"password": "wrong"}).status_code == 401
    auth = headers(client)
    sample = client.get("/api/demo/sample", headers=auth).json()
    response = client.post("/api/analyze", headers=auth, json=sample)
    assert response.status_code == 200
    data = response.json()
    assert data["confidence"] is None
    assert data["report"]["metadata"] == sample["metadata"]
    assert data["report"]["classifier_model_ids"]
    assert data["report"]["clause_status_inventory"]
    assert data["engine"] == "offline-deterministic"
    assert len(data["findings"]) == len(data["report"]["findings"])
    evidence = {item["evidence_id"] for item in data["report"]["evidence"]}
    assert all(set(finding["citation_ids"]) <= evidence for finding in data["report"]["findings"])
    assert len(client.get("/api/sources", headers=auth).json()) == 20


def test_lens_requires_context_and_meaningful_input(client):
    auth = headers(client)
    assert client.post("/api/analyze", headers=auth, json={"text": "x" * 100}).status_code == 422
    payload = {"text": " " * 100, "metadata": sample_metadata().model_dump(mode="json")}
    assert client.post("/api/analyze", headers=auth, json=payload).status_code == 422


def test_model_mode_does_not_fall_back_without_model():
    model_client = TestClient(
        create_app(
            Settings(
                model_review_enabled=True,
                bedrock_enabled=True,
                classifier_model_dir="missing-model",
            )
        )
    )
    auth = headers(model_client)
    sample = model_client.get("/api/demo/sample", headers=auth).json()
    result = model_client.post("/api/analyze", headers=auth, json=sample)
    assert result.status_code == 503
    assert "trained classifier" in result.json()["detail"]


def test_authenticated_docx_upload_reuses_native_parser(client):
    auth = headers(client)
    response = client.post(
        "/api/review-upload",
        headers=auth,
        data={"metadata_json": sample_metadata().model_dump_json()},
        files={"file": ("sample.docx", sample_contract_bytes())},
    )
    assert response.status_code == 200
    assert response.json()["report"]["filename"] == "sample.docx"
    rejected = client.post(
        "/api/review-upload",
        headers=auth,
        data={"metadata_json": sample_metadata().model_dump_json()},
        files={"file": ("sample.exe", b"bad")},
    )
    assert rejected.status_code == 415
