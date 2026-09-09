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
    from app.workspace_export import WorkspaceExport, record_digest

    WorkspaceExport.model_validate(exported)
    assert exported["schema_version"] == "1.1"
    assert exported["record_sha256"] == record_digest(exported["record"])
    assert exported["record_sha256"] != record_digest(exported["record"] | {"note": "Changed"})
    assert exported["sources"][0]["document_id"] == doc["id"]
    assert exported["sources"][0]["role"] == "context"
    assert exported["sources"][0]["security_label"] == "demo"
    assert (
        client.get("/api/workspace/export-schema").json()["properties"]["schema_version"]["const"]
        == "1.1"
    )
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
        "export-schema",
    ):
        assert client.get(f"/api/workspace/{path}").status_code == 401
    assert client.get("/ui/").status_code == 404


def test_readiness_matches_export_and_stale_reviews_can_be_returned(setup):
    client, _, _, _ = setup
    doc = add(client)
    review = client.post(f"/api/workspace/documents/{doc['id']}/review").json()
    status = client.get("/api/workspace/reviews").json()[0]["readiness"]
    assert status["can_approve"] and not status["can_export"]
    assert not status["blockers"]
    client.put(
        f"/api/workspace/documents/{doc['id']}/metadata",
        json={"metadata": doc["metadata"] | {"agency": "Changed"}, "revision": doc["revision"]},
    )
    status = client.get("/api/workspace/reviews").json()[0]["readiness"]
    assert not status["can_approve"]
    assert status["blockers"][0]["code"] == "metadata_changed"
    url = f"/api/workspace/records/{review['id']}/decision"
    body = {"note": "Acquisition details need a fresh review", "revision": review["revision"]}
    assert client.post(url, json=body | {"decision": "approved"}).status_code == 409
    assert client.post(url, json=body | {"decision": "rejected"}).status_code == 200


def test_context_version_is_checked_even_when_answer_only_cites_another_source(setup):
    client, _, _, principal = setup
    context = add(client)
    reference = add(
        client, "Original source says North Company warranty is twelve months. " * 3, "reference"
    )
    question = client.post(
        "/api/workspace/questions",
        json={"query": "What is the warranty?", "document_id": context["id"]},
    ).json()
    assert question["document_version"] == context["version"]
    db = client.app.state.workspace_store
    # Model output may cite only the reference, although the selected contract is context.
    evidence = [e for e in question["response"]["evidence"] if e["document_id"] == reference["id"]]
    assert evidence
    question["response"]["evidence"] = evidence
    question["response"]["statements"] = [
        {
            "text": "The warranty is twelve months.",
            "grounding_status": "verified",
            "citation_ids": [evidence[0]["evidence_id"]],
        }
    ]
    db.save(principal, "question", question, question["id"], question["revision"])
    record = client.post(
        "/api/workspace/structured-records",
        json={
            "question_id": question["id"],
            "title": "Warranty memo",
        },
    ).json()
    assert record["document_version"] == context["version"]
    assert client.get("/api/workspace/structured-records").json()[0]["readiness"]["can_approve"]
    context["version"] = "new-version"
    db.save(principal, "document", context, context["id"], context["revision"])
    status = client.get("/api/workspace/structured-records").json()[0]["readiness"]
    assert not status["can_approve"]
    assert status["blockers"][0]["code"] == "source_changed"
    result = client.post(
        f"/api/workspace/records/{record['id']}/decision",
        json={
            "decision": "approved",
            "note": "Checked the evidence",
            "revision": record["revision"],
        },
    )
    assert result.status_code == 409


@pytest.mark.parametrize("citations", [[], ["missing-citation"]])
def test_verified_label_alone_does_not_allow_export(setup, citations):
    client, _, _, principal = setup
    db = client.app.state.workspace_store
    record = db.save(
        principal,
        "structured-record",
        {
            "title": "Invalid provenance",
            "decision": "approved",
            "document_id": None,
            "content": {
                "evidence": [],
                "statements": [
                    {
                        "text": "Unsupported assertion",
                        "grounding_status": "verified",
                        "citation_ids": citations,
                    }
                ],
            },
        },
    )
    response = client.get(f"/api/workspace/records/{record['id']}/export")
    assert response.status_code == 409
    status = client.get("/api/workspace/structured-records").json()[0]["readiness"]
    assert status["blockers"][0]["code"] == "unresolved_citations"


def test_readiness_does_not_hydrate_source_bodies(setup):
    client, _, _, principal = setup
    from app.workspace_readiness import ReadinessEvaluator

    doc = add(client)
    db = client.app.state.workspace_store
    entity = {"kind": "entity", "links": [{"document_id": doc["id"], "version": doc["version"]}]}
    evaluator = ReadinessEvaluator(db, principal)
    with patch.object(db, "get", wraps=db.get) as get:
        assert evaluator.evaluate(entity)["can_approve"]
        assert evaluator.evaluate(entity)["can_approve"]
    get.assert_called_once_with(principal, doc["id"], hydrate=False)


def test_source_readiness_distinguishes_indexed_text_from_current_access(setup):
    client, _, _, principal = setup
    db = client.app.state.workspace_store
    doc = add(client)
    # A record may retain indexed text after its originating connection disappears.
    doc.update(source_provider="sharepoint", connection_id="missing-connection")
    db.save(principal, "document", doc, doc["id"], doc["revision"])
    listed = client.get("/api/workspace/documents").json()[0]
    assert listed["status"] == "ready" and listed["available"] is False
    detail = client.get(f"/api/workspace/documents/{doc['id']}").json()
    assert detail["available"] is False and "parsed" not in detail
    assert detail["text"] == doc["text"]
    result = client.post(f"/api/workspace/documents/{doc['id']}/review")
    assert result.status_code == 409
    entity = db.save(
        principal,
        "entity",
        {
            "links": [{"document_id": doc["id"], "version": doc["version"]}],
            "decision": "draft",
        },
    )
    status = client.get("/api/workspace/entities").json()[0]["readiness"]
    assert status["blockers"][0]["code"] == "source_not_ready"
    assert not status["can_approve"]
    assert (
        client.post(
            f"/api/workspace/records/{entity['id']}/decision",
            json={
                "decision": "approved",
                "note": "Checked",
                "revision": entity["revision"],
            },
        ).status_code
        == 409
    )


def test_whitespace_inputs_do_not_start_generation_or_create_approvals(setup):
    client, _, _, _ = setup
    assert (
        client.post(
            "/api/workspace/documents",
            json={
                "title": "Empty input",
                "text": " " * 100,
            },
        ).status_code
        == 422
    )
    assert client.post("/api/workspace/questions", json={"query": "   "}).status_code == 422
    doc = add(client)
    review = client.post(f"/api/workspace/documents/{doc['id']}/review").json()
    assert (
        client.post(
            f"/api/workspace/records/{review['id']}/decision",
            json={
                "decision": "approved",
                "note": "   ",
                "revision": review["revision"],
            },
        ).status_code
        == 422
    )


def test_missing_and_foreign_evidence_is_reported_without_leaking_source_details(setup):
    client, _, _, principal = setup
    db = client.app.state.workspace_store
    foreign = db.save(
        PrincipalContext(subject="bob", security_domain="demo"),
        "document",
        {
            "title": "Other reviewers private title",
            "status": "ready",
            "version": "v1",
        },
    )
    for record_id in (foreign["id"], "missing-document"):
        db.save(
            principal,
            "entity",
            {
                "decision": "draft",
                "links": [{"document_id": record_id, "version": "v1"}],
            },
        )
    response = client.get("/api/workspace/entities")
    assert response.status_code == 200
    assert "Other reviewers private title" not in response.text
    assert all(
        e["readiness"]["blockers"][0]["code"] == "source_unavailable" for e in response.json()
    )


def test_export_does_not_merge_unrelated_legacy_citations_without_document_ids(setup):
    client, _, _, principal = setup
    from app.adapters import EVIDENCE_CATALOG

    evidence = [e.model_dump(mode="json") for e in EVIDENCE_CATALOG[:2]]
    record = client.app.state.workspace_store.save(
        principal,
        "structured-record",
        {
            "title": "Legacy reviewed record",
            "decision": "approved",
            "document_id": None,
            "content": {
                "evidence": evidence,
                "statements": [
                    {
                        "text": "Check both sources",
                        "grounding_status": "verified",
                        "citation_ids": [e["evidence_id"] for e in evidence],
                    }
                ],
            },
        },
    )
    response = client.get(f"/api/workspace/records/{record['id']}/export")
    assert response.status_code == 200
    sources = response.json()["sources"]
    assert len(sources) == 2
    assert {s["title"] for s in sources} == {e["title"] for e in evidence}
    assert all(s["document_id"] == "" and s["version"] == "" for s in sources)
