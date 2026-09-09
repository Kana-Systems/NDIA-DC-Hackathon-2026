"""Transient Graph failures recover without importing partial or unauthorized content."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.config import Settings
from app.models import PrincipalContext
from app.workspace import parsed_text, save_document, sync_connection
from app.workspace_store import WorkspaceStore
from ingestion.graph import GraphClient


def response(status=200, body=b"", **headers):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    result.headers.update(headers)
    result.url = "https://tenant.sharepoint.com/file?secret=DO_NOT_LOG"
    return result


def client():
    graph = GraphClient(
        client_id="test", client_secret="test", tenant_id="test", session=MagicMock()
    )
    graph._access_token = "test-token"
    return graph


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_graph_listing_retries_transient_status_and_honors_retry_after(status):
    graph = client()
    graph.session.get.side_effect = [
        response(status, **{"Retry-After": "3"}),
        response(body=b'{"value": []}'),
    ]
    with patch("ingestion.graph.time.sleep") as sleep:
        assert graph.get_json("/drives/test/root/children") == {"value": []}
    sleep.assert_called_once_with(3)
    assert graph.session.get.call_count == 2


@pytest.mark.parametrize(
    "status,header,count", [(403, "1", 1), (404, "1", 1), (429, "120", 1), (503, "", 3)]
)
def test_permanent_errors_and_retry_budget_stop_requests(status, header, count):
    graph = client()
    graph.session.get.side_effect = lambda *a, **k: response(status, **{"Retry-After": header})
    with patch("ingestion.graph.time.sleep") as sleep, pytest.raises(requests.HTTPError):
        graph.get_json("/drives/test/root/children")
    assert graph.session.get.call_count == count
    assert sleep.call_count == count - 1


def test_graph_refreshes_expired_token_once():
    graph = client()
    graph.session.get.side_effect = [response(401), response(body=b"{}")]
    graph.session.post.return_value = response(body=b'{"access_token":"refreshed"}')
    assert graph.get_json("/drives/test") == {}
    assert graph.session.get.call_args.kwargs["headers"] == {"Authorization": "Bearer refreshed"}


@pytest.mark.parametrize("failure", ["throttle", "interrupted", "timeout"])
def test_download_restarts_with_fresh_url_and_discards_partial_body(failure):
    graph = client()
    graph.session.get.side_effect = [
        response(302, Location="https://tenant.sharepoint.com/first"),
        response(302, Location="https://tenant.sharepoint.com/second"),
    ]
    if failure == "throttle":
        failed = response(429, **{"Retry-After": "2"})
    elif failure == "timeout":
        failed = requests.Timeout("DO_NOT_LOG")
    else:
        failed = response()

        def chunks(*args):
            yield b"partial"
            raise requests.exceptions.ChunkedEncodingError("interrupted")

        failed.iter_content = chunks
    with (
        patch("requests.get", side_effect=[failed, response(body=b"complete")]) as get,
        patch("ingestion.graph.time.sleep") as sleep,
    ):
        assert graph.download("/drives/test/items/file/content", 20) == b"complete"
    assert graph.session.get.call_count == 2
    assert [c.args[0] for c in get.call_args_list] == [
        "https://tenant.sharepoint.com/first",
        "https://tenant.sharepoint.com/second",
    ]
    assert all(
        "headers" not in c.kwargs and not c.kwargs["allow_redirects"] for c in get.call_args_list
    )
    sleep.assert_called_once()


def test_size_and_untrusted_host_fail_without_retries():
    graph = client()
    for result in [
        response(body=b"too big"),
        response(302, Location="https://attacker.invalid/file"),
    ]:
        graph.session.get.return_value = result
        with (
            patch("ingestion.graph.time.sleep") as sleep,
            patch("requests.get") as get,
            pytest.raises(ValueError),
        ):
            graph.download("/drives/test/items/file/content", 3)
        sleep.assert_not_called()
        get.assert_not_called()


@pytest.mark.parametrize(
    "failure,expected",
    [
        (requests.HTTPError(response=response(403)), "denied access"),
        (requests.HTTPError(response=response(429)), "temporarily unavailable"),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"), "UTF-8"),
        (ValueError("Document exceeds upload size limit"), "size limit"),
        (ValueError("Document parsing failed in isolated worker"), "could not be parsed"),
    ],
)
def test_sync_reports_filename_and_safe_reason_without_removing_failed_document(
    tmp_path, caplog, failure, expected
):
    settings = Settings(
        workspace_password="test-password",
        bedrock_enabled=False,
        workspace_db_path=str(tmp_path / "workspace.sqlite"),
    )
    principal = PrincipalContext(subject="test", security_domain="demo")
    db = WorkspaceStore(settings)
    connection = db.save(
        principal, "connection", {"provider": "sharepoint", "folder": "", "category": "reference"}
    )
    old = save_document(
        db,
        principal,
        parsed_text("Agreement.txt", "Original source. " * 10),
        "reference",
        None,
        connection_id=connection["id"],
        source_key="opaque-id",
    )
    graph = MagicMock()
    graph.get_json.return_value = {
        "value": [{"id": "opaque-id", "name": "Agreement.txt", "file": {}}]
    }
    graph.download.side_effect = failure
    if "parsed" in expected:
        graph.get_json.return_value["value"][0]["name"] = "Agreement.docx"
        graph.download.side_effect = None
        graph.download.return_value = b"invalid docx fixture"
    with (
        patch("app.sharepoint.check_connection", return_value=(graph, {"drive_id": "test"}, {})),
        patch("app.sharepoint.DocumentParser.parse_isolated", side_effect=failure),
    ):
        sync_connection(db, principal, connection["id"], settings)
    updated = db.get(principal, connection["id"])
    assert updated["status"] == "partial"
    assert updated["errors"][0]["file"].startswith("Agreement.")
    assert expected in updated["errors"][0]["message"]
    assert updated["counts"]["removed"] == 0
    assert db.get(principal, old["id"])["status"] == "source-error"
    assert not db.get(principal, old["id"]).get("deleted")
    assert "DO_NOT_LOG" not in json.dumps(updated["errors"]) + caplog.text


def test_retry_after_http_date_and_invalid_header():
    from datetime import UTC, datetime

    now = datetime(2026, 9, 9, 19, 0, 0, tzinfo=UTC)
    for header, expected in [("Wed, 09 Sep 2026 19:00:05 GMT", 5), ("invalid", 1)]:
        graph = client()
        graph.session.get.side_effect = [
            response(503, **{"Retry-After": header}),
            response(body=b"{}"),
        ]
        with (
            patch("ingestion.graph.datetime") as clock,
            patch("ingestion.graph.time.sleep") as sleep,
        ):
            clock.now.return_value = now
            assert graph.get_json("/drives/test") == {}
        sleep.assert_called_once_with(expected)
