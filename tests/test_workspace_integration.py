"""Cloud adapters tested with isolated test doubles; no fabricated deployed data."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.models import PrincipalContext
from app.sharepoint import configured_client, source_files
from app.workspace import parsed_text, save_document
from app.workspace_search import WorkspaceSearch
from app.workspace_store import WorkspaceStore
from app.workspace_sync import claim
from ingestion.graph import GraphClient


@pytest.fixture
def context(tmp_path):
    return (
        Settings(
            workspace_password="pytest-only-password",
            bedrock_enabled=False,
            workspace_db_path=str(tmp_path / "workspace.sqlite"),
        ),
        PrincipalContext(subject="alice", security_domain="demo"),
    )


def test_s3_body_is_not_duplicated_in_database_and_hydration_is_scoped(context):
    settings, principal = context
    objects = {}
    s3 = MagicMock()
    s3.put_object.side_effect = lambda **args: objects.update({args["Key"]: args["Body"]})
    s3.get_object.side_effect = lambda **args: {"Body": io.BytesIO(objects[args["Key"]])}
    with patch("boto3.client", return_value=s3):
        db = WorkspaceStore(settings.model_copy(update={"workspace_source_bucket": "test-bucket"}))
    doc = parsed_text("Input.txt", "Original unit-test source. " * 10)
    saved = save_document(db, principal, doc, "reference", raw_bytes=b"Original bytes")
    listed = db.list(principal, "document")[0]
    assert "parsed" not in listed and "text" not in listed
    assert listed["body_key"] and listed["original_key"]
    assert objects[listed["original_key"]] == b"Original bytes"
    assert db.get(principal, saved["id"])["text"] == doc.text
    with pytest.raises(HTTPException) as error:
        db.get(PrincipalContext(subject="bob"), saved["id"])
    assert error.value.status_code == 404
    assert all(
        call.kwargs["ServerSideEncryption"] == "aws:kms" for call in s3.put_object.call_args_list
    )


def test_index_failure_is_visible_and_retry_returns_ready(context):
    from app.workspace import index_document

    settings, principal = context
    db = WorkspaceStore(settings)
    db.search = MagicMock()
    db.search.index_document.side_effect = RuntimeError("index unavailable")
    saved = save_document(db, principal, parsed_text("file", "test passage " * 10), "reference")
    assert saved["status"] == "index-failed"
    assert db.get(principal, saved["id"])["text"]
    db.search.index_document.side_effect = None
    assert index_document(db, principal, saved)["status"] == "ready"


def test_cloud_ingestion_keeps_whole_long_documents_and_bounds_index_batches(context):
    settings, principal = context
    with patch("boto3.client", return_value=MagicMock()):
        db = WorkspaceStore(settings.model_copy(update={"workspace_source_bucket": "test-bucket"}))
    text = "Complete original fixture text. " * 1400
    doc = save_document(db, principal, parsed_text("long-source.txt", text), "reference")
    assert len(doc["text"]) > 30000 and doc["text"] == text
    assert "text" not in db.list(principal, "document")[0]
    client = MagicMock()
    client.request.return_value = {"errors": False}
    search = WorkspaceSearch(settings, client=client, embed=lambda _: [0.1] * 1024)
    search.index_document(doc)
    batches = [c for c in client.request.call_args_list if c.args[0] == "POST"]
    indexed = [
        json.loads(line)["text"] for call in batches for line in call.args[2].splitlines()[1::2]
    ]
    assert len(batches) > 1
    assert "".join(indexed) == text
    assert all(len(c.args[2].splitlines()) <= 50 for c in batches)


def test_hybrid_queries_filter_owner_and_recheck_versions(context):
    settings, principal = context
    db = WorkspaceStore(settings)
    doc = save_document(db, principal, parsed_text("Policy", "Policy text " * 10), "policy")
    source = {
        "evidence_id": "workspace:valid",
        "document_id": doc["id"],
        "version": doc["version"],
        "owner": "alice",
        "security_domain": "demo",
        "text": "Policy text",
    }
    hits = [
        {"_source": source},
        {"_source": {**source, "evidence_id": "workspace:foreign", "owner": "bob"}},
        {"_source": {**source, "evidence_id": "workspace:stale", "version": "old"}},
    ]
    client = MagicMock()
    client.request.return_value = {"hits": {"hits": hits}}
    search = WorkspaceSearch(
        settings.model_copy(update={"opensearch_vector_enabled": True}),
        client=client,
        embed=lambda _: [0.1] * 1024,
    )
    result = search.retrieve(["policy"], principal, db, references_only=True)
    assert [e.evidence_id for e in result] == ["workspace:valid"]
    calls = [c for c in client.request.call_args_list if c.args[0] == "POST"]
    assert len(calls) == 2
    for call in calls:
        query = call.args[2]
        assert '"owner": "alice"' in query and '"security_domain": "demo"' in query
        assert '"must_not"' in query


def test_graph_rejects_untrusted_token_and_pagination_urls_before_auth():
    session = MagicMock()
    with pytest.raises(ValueError):
        GraphClient(
            client_id="id",
            client_secret="secret",
            tenant_id="tenant",
            token_url="https://attacker.invalid/token",
            session=session,
        )
    graph = GraphClient(client_id="id", client_secret="secret", tenant_id="tenant", session=session)
    with pytest.raises(ValueError):
        graph.get_json("https://attacker.invalid/page")
    session.post.assert_not_called()
    session.get.assert_not_called()


def test_graph_national_cloud_selects_matching_token_authority():
    graph = GraphClient(
        client_id="id",
        client_secret="secret",
        tenant_id="tenant",
        graph_base_url="https://graph.microsoft.us/v1.0",
    )
    assert graph.token_url == "https://login.microsoftonline.us/tenant/oauth2/v2.0/token"


def test_download_is_bounded_and_redirect_does_not_forward_bearer():
    graph = GraphClient(client_id="id", client_secret="secret", tenant_id="tenant")
    redirect = MagicMock(
        status_code=302, headers={"Location": "https://tenant.sharepoint.com/download"}
    )
    downloaded = MagicMock(status_code=200, headers={})
    downloaded.iter_content.return_value = [b"abc", b"def"]
    with (
        patch.object(graph, "_get", return_value=redirect),
        patch("requests.get", return_value=downloaded) as get,
    ):
        assert graph.download("/drives/d/items/f/content", 6) == b"abcdef"
        assert "headers" not in get.call_args.kwargs
    downloaded.iter_content.return_value = [b"0123456789"]
    with patch.object(graph, "_get", return_value=downloaded), pytest.raises(ValueError):
        graph.download("/drives/d/items/f/content", 6)


def test_sharepoint_requires_explicit_workspace_library_approval(context):
    settings, principal = context
    config = {
        "tenant_id": "tenant",
        "client_id": "id",
        "client_secret": "test-secret",
        "site_id": "site",
        "drive_id": "drive",
        "security_domain": "demo",
        "allowed_workspace_subjects": ["alice"],
    }
    secret = MagicMock()
    secret.get_secret_value.return_value = {"SecretString": json.dumps(config)}
    with patch("boto3.client", return_value=secret), pytest.raises(HTTPException) as error:
        configured_client(
            settings.model_copy(update={"graph_connector_secret_arn": "test-arn"}), principal
        )
    assert error.value.status_code == 403


def test_sharepoint_pages_folders_real_parser_and_shortcut_exclusion(context):
    from docx import Document

    settings, principal = context
    document = Document()
    document.add_paragraph("Original isolated DOCX fixture, not runtime data.")
    raw = io.BytesIO()
    document.save(raw)
    graph = MagicMock()
    graph.get_json.side_effect = [
        {
            "value": [{"id": "folder", "folder": {}}, {"id": "shortcut", "remoteItem": {}}],
            "@odata.nextLink": "https://graph.microsoft.com/v1.0/page2",
        },
        {"value": [{"id": "text", "name": "source.txt", "file": {}, "eTag": "v2"}]},
        {"value": [{"id": "docx", "name": "source.docx", "file": {}, "eTag": "v1"}]},
    ]
    graph.download.side_effect = [b"Original temporary test source " * 4, raw.getvalue()]
    with patch("app.sharepoint.check_connection", return_value=(graph, {"drive_id": "drive"}, {})):
        result = list(source_files({"folder": ""}, settings, principal))
    assert [r[0] for r in result] == ["text", "docx"]
    assert all(r[2] is None for r in result)
    assert "DOCX fixture" in result[1][1].text
    assert result[1][3]["raw_bytes"] == raw.getvalue()
    assert graph.download.call_count == 2


def test_sync_lease_prevents_concurrent_sync_and_allows_restart_recovery(context):
    settings, principal = context
    db = WorkspaceStore(settings)
    connection = db.save(principal, "connection", {"provider": "sharepoint", "status": "idle"})
    claimed = claim(db, principal, connection)
    with pytest.raises(HTTPException) as error:
        claim(db, principal, claimed)
    assert error.value.status_code == 409
    claimed["sync_started_at"] = "2000-01-01T00:00:00+00:00"
    claimed = db.save(principal, "connection", claimed, claimed["id"], claimed["revision"])
    assert claim(db, principal, claimed)["sync_lease"] != claimed["sync_lease"]


def test_sharepoint_sources_fail_closed_when_sync_fails_or_is_stale(context):
    from app.workspace_store import now

    settings, principal = context
    db = WorkspaceStore(settings)
    connection = db.save(principal, "connection", {"status": "complete", "last_sync": now()})
    doc = {"source_provider": "sharepoint", "connection_id": connection["id"], "status": "ready"}
    assert db.source_available(principal, doc)
    connection["status"] = "failed"
    db.save(principal, "connection", connection, connection["id"], connection["revision"])
    assert not db.source_available(principal, doc)
    connection = db.get(principal, connection["id"])
    connection.update(status="complete", last_sync="2000-01-01T00:00:00+00:00")
    db.save(principal, "connection", connection, connection["id"], connection["revision"])
    assert not db.source_available(principal, doc)


def test_federal_index_ranks_in_opensearch_but_verifies_original_text(context, tmp_path):
    import sqlite3

    settings, _ = context
    path = tmp_path / "official-unit-test.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        db.execute("INSERT INTO metadata VALUES (?, ?)", ("manifest", json.dumps({"chunks": 1})))
        db.execute(
            "CREATE TABLE chunks (evidence_id TEXT, document_id TEXT, version TEXT, heading TEXT, "
            "text TEXT, url TEXT, authority TEXT, locator TEXT)"
        )
        db.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "e1",
                "d1",
                "v1",
                "Test heading",
                "Canonical fixture text",
                "https://example.org",
                "FAR",
                "test",
            ),
        )
    client = MagicMock()
    client.request.side_effect = [
        {},
        {"count": 0},
        {"errors": False},
        {},
        {"count": 1},
        {
            "hits": {
                "hits": [
                    {"_source": {"evidence_id": "e1", "text": "Do not trust this index text"}},
                    {"_source": {"evidence_id": "unknown"}},
                ]
            }
        },
    ]
    embed = MagicMock()
    search = WorkspaceSearch(
        settings.model_copy(update={"local_corpus_path": str(path)}), client, embed
    )
    assert search.index_federal() == 1
    result = search.retrieve_federal(["test"])
    assert len(result) == 1 and result[0].excerpt == "Canonical fixture text"
    embed.assert_not_called()  # Official corpus indexing does not incur mass embedding calls.
    query = client.request.call_args.args[2]
    assert '"owner": "official"' in query
    assert '"security_domain": "public"' in query


def test_explicit_far_and_dfars_identifiers_outrank_generic_question_words(context):
    settings, _ = context
    client = MagicMock()
    client.request.side_effect = RuntimeError("stop after inspecting query")
    search = WorkspaceSearch(settings, client, MagicMock())
    with (
        patch.object(search, "federal_index", return_value=("test-official", {})),
        pytest.raises(RuntimeError, match="stop after inspecting query"),
    ):
        search.retrieve_federal(
            ["Which official passages mention FAR 52.249-2 or DFARS 252.204-7012?"]
        )
    query = json.loads(client.request.call_args.args[2])["query"]["bool"]
    exact = query["should"][1]["constant_score"]
    assert exact["filter"]["terms"]["document_id"] == ["FAR:52.249-2", "DFARS:252.204-7012"]
    assert exact["boost"] == 1000
    assert query["filter"] == [
        {"term": {"owner": "official"}},
        {"term": {"security_domain": "public"}},
    ]
