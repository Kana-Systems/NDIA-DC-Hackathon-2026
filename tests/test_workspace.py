"""Connected workflow, persistence, version gates, and access isolation."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.lens import require_reviewer
from app.main import create_app
from app.models import PrincipalContext
from app.sample import sample_metadata
from app.workspace import WorkspaceGeneration, WorkspaceRetrieval
from app.workspace_store import WorkspaceStore


def test_dynamodb_persistence_pagination_and_conflict(setup):
    import json

    from botocore.exceptions import ClientError

    _, settings, _, principal = setup
    table = MagicMock()
    table.get_item.return_value = {}
    with patch("boto3.resource") as resource:
        resource.return_value.Table.return_value = table
        db = WorkspaceStore(settings.model_copy(update={"workspace_table": "lens-test"}))
    record = db.save(principal, "event", {"action": "test"})
    item = table.put_item.call_args.kwargs["Item"]
    assert item["scope"] == '["demo","alice"]'
    assert item["revision"] == 1
    assert json.loads(item["data"])["id"] == record["id"]
    table.get_item.return_value = {"Item": item}
    assert db.get(principal, record["id"]) == record
    table.query.side_effect = [
        {"Items": [item], "LastEvaluatedKey": {"scope": item["scope"], "id": record["id"]}},
        {"Items": []},
    ]
    assert db.list(principal, "event") == [record]
    assert table.query.call_count == 2
    table.put_item.side_effect = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem"
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        db.save(principal, "event", record, record["id"], 1)
    assert error.value.status_code == 409


def test_reference_change_blocks_approved_entity_and_record(setup):
    client, _, _, principal = setup
    doc = add(client)
    entity = client.post(
        "/api/workspace/entities",
        json={
            "name": "North Company",
            "entity_type": "vendor",
            "document_id": doc["id"],
            "excerpt": "North Company",
        },
    ).json()
    saved = client.app.state.workspace_store.get(principal, doc["id"])
    saved["version"] = "changed"
    client.app.state.workspace_store.save(principal, "document", saved, doc["id"])
    result = client.post(
        f"/api/workspace/records/{entity['id']}/decision",
        json={
            "decision": "approved",
            "note": "Checked sources",
            "revision": entity["revision"],
        },
    )
    assert result.status_code == 409


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "import"
    root.mkdir()
    settings = Settings(
        workspace_password="pytest-only-password",
        bedrock_enabled=False,
        model_review_enabled=False,
        workspace_db_path=str(tmp_path / "lens.sqlite"),
        workspace_import_root=str(root),
        local_corpus_path="",
    )
    app = create_app(settings)
    alice = PrincipalContext(subject="alice", groups=["contract-reviewers"], security_domain="demo")
    app.dependency_overrides[require_reviewer] = lambda: alice
    return TestClient(app), settings, root, alice


def add(client, text=None, category="contract"):
    result = client.post(
        "/api/workspace/documents",
        json={
            "title": "Original supplied contract",
            "category": category,
            "text": text or ("The agreement names North Company as the supplier. " * 4),
            "metadata": sample_metadata().model_dump(mode="json"),
        },
    )
    assert result.status_code == 200, result.text
    return result.json()


def test_persisted_document_is_isolated_by_user_and_domain(setup):
    client, settings, _, alice = setup
    doc = add(client)
    restarted = create_app(settings)
    restarted.dependency_overrides[require_reviewer] = lambda: alice
    assert TestClient(restarted).get("/api/workspace/documents").json()[0]["id"] == doc["id"]
    for other in (
        PrincipalContext(subject="bob"),
        PrincipalContext(subject="alice", security_domain="other"),
    ):
        client.app.dependency_overrides[require_reviewer] = lambda other=other: other
        assert client.get("/api/workspace/documents").json() == []
        assert client.get(f"/api/workspace/documents/{doc['id']}").status_code == 404


def test_review_approval_export_and_stale_metadata_gate(setup):
    client, _, _, _ = setup
    doc = add(client)
    response = client.post(f"/api/workspace/documents/{doc['id']}/review")
    assert response.status_code == 200, response.text
    review = response.json()
    assert client.get(f"/api/workspace/records/{review['id']}/export").status_code == 409
    approved = client.post(
        f"/api/workspace/records/{review['id']}/decision",
        json={
            "decision": "approved",
            "note": "Reviewed original text and limitations",
            "revision": 1,
        },
    )
    assert approved.status_code == 200
    exported = client.get(f"/api/workspace/records/{review['id']}/export").json()
    assert exported["external_write_performed"] is False
    metadata = doc["metadata"] | {"agency": "Changed agency"}
    assert (
        client.put(
            f"/api/workspace/documents/{doc['id']}/metadata",
            json={
                "metadata": metadata,
                "revision": doc["revision"],
            },
        ).status_code
        == 200
    )
    assert client.get(f"/api/workspace/records/{review['id']}/export").status_code == 409
    assert (
        client.put(
            f"/api/workspace/documents/{doc['id']}/metadata",
            json={
                "metadata": metadata,
                "revision": doc["revision"],
            },
        ).status_code
        == 409
    )


def test_context_questions_and_durable_records_do_not_use_fixtures(setup):
    client, settings, _, alice = setup
    doc = add(
        client, "North Company supplier agreement. Original warranty lasts twelve months. " * 3
    )
    other = add(client, "A confidential second contract has a 999-month warranty. " * 3)
    retrieval = WorkspaceRetrieval(client.app.state.workspace_store, alice, settings, doc["id"])
    evidence = retrieval.retrieve(["warranty"])
    assert evidence and all(e.document_id != other["id"] for e in evidence)
    q = client.post(
        "/api/workspace/questions",
        json={
            "query": "What warranty does this contract specify?",
            "document_id": doc["id"],
        },
    )
    assert q.status_code == 200, q.text
    assert q.json()["response"]["synthesis_mode"] == "offline-extractive"
    assert "999" not in q.json()["response"]["answer"]
    assert all(e["evidence_id"].startswith("workspace:") for e in q.json()["response"]["evidence"])
    record = client.post(
        "/api/workspace/structured-records",
        json={
            "title": "Warranty review",
            "question_id": q.json()["id"],
        },
    ).json()
    assert len(client.get("/api/workspace/structured-records").json()) == 1
    assert client.get(f"/api/workspace/records/{record['id']}/export").status_code == 409


def test_live_generation_errors_do_not_silently_fall_back(setup):
    _, settings, _, _ = setup
    generator = WorkspaceGeneration(settings.model_copy(update={"bedrock_enabled": True}))
    with (
        patch.object(generator, "_remote_synthesis", side_effect=RuntimeError("unavailable")),
        pytest.raises(RuntimeError, match="unavailable"),
    ):
        generator._synthesize(None, [])


def test_source_sync_add_update_delete_and_error_reporting(setup):
    client, _, root, _ = setup
    # Test-only temporary inputs; not runtime datasets or hackathon submission data.
    for index in range(101):
        (root / f"document-{index}.txt").write_text(f"Original test file {index}. " * 10)
    response = client.post(
        "/api/workspace/connections",
        json={
            "name": "Approved folder",
            "provider": "shared-folder",
            "folder": "",
            "category": "reference",
        },
    )
    connection = response.json()
    assert client.post(f"/api/workspace/connections/{connection['id']}/sync").status_code == 200
    result = client.get("/api/workspace/connections").json()["connections"][0]
    assert result["counts"]["added"] == 101
    assert len(client.get("/api/workspace/documents").json()) == 101
    (root / "document-0.txt").write_text("Changed version of the actual input. " * 10)
    (root / "document-1.txt").unlink()
    (root / "broken.txt").write_bytes(b"\xff\xff")
    assert client.post(f"/api/workspace/connections/{connection['id']}/sync").status_code == 200
    result = client.get("/api/workspace/connections").json()["connections"][0]
    assert result["counts"] == {"added": 0, "updated": 1, "unchanged": 99, "removed": 1}
    assert result["errors"] == [{"file": "broken.txt", "message": "Document could not be parsed"}]
    assert result["status"] == "partial"


def test_import_root_escape_is_rejected(setup):
    client, _, root, _ = setup
    (root / "escape").symlink_to(root.parent, target_is_directory=True)
    for folder in ("../", "escape", "/etc"):
        response = client.post(
            "/api/workspace/connections",
            json={
                "name": "Not approved",
                "provider": "shared-folder",
                "folder": folder,
            },
        )
        assert response.status_code == 422


def test_entities_require_exact_source_and_return_to_draft_on_change(setup):
    client, _, _, _ = setup
    doc = add(client)
    payload = {
        "name": "North Company",
        "entity_type": "vendor",
        "document_id": doc["id"],
        "excerpt": "Not present anywhere",
    }
    assert client.post("/api/workspace/entities", json=payload).status_code == 422
    payload["excerpt"] = "North Company"
    first = client.post("/api/workspace/entities", json=payload).json()
    assert (
        client.post(
            f"/api/workspace/records/{first['id']}/decision",
            json={
                "decision": "approved",
                "note": "Verified the source excerpt",
                "revision": first["revision"],
            },
        ).status_code
        == 200
    )
    payload["name"] = "north company"
    second = client.post("/api/workspace/entities", json=payload).json()
    assert second["id"] == first["id"]
    assert second["decision"] == "draft"


def test_auth_required_for_all_workspace_reads():
    client = TestClient(create_app(Settings()))
    for path in (
        "documents",
        "reviews",
        "connections",
        "questions",
        "entities",
        "events",
        "library",
        "structured-records",
    ):
        assert client.get(f"/api/workspace/{path}").status_code == 401
    assert client.get("/ui/").status_code == 404
