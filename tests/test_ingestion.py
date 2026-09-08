import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ingestion.aws_store import AwsIngestionStore
from ingestion.chunking import chunk_text, normalize_document
from ingestion.cli import main as cli_main
from ingestion.connectors import Connector, FilesystemConnector
from ingestion.fixtures import FIXTURE_DOCUMENT_COUNT, generate_fixture_corpus
from ingestion.graph import GraphClient, GraphDeltaConnector
from ingestion.models import ChangeEvent, ChangeKind, SourceDocument
from ingestion.pipeline import ingest
from ingestion.store import InMemoryIngestionStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "enterprise"


class FakeResponse:
    def __init__(self, *, payload=None, content=b"", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP status {self.status_code}")


class FakeGraphSession:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post(self, url, *, data, timeout):
        self.posts.append((url, data, timeout))
        return FakeResponse(payload={"access_token": "test-token"})

    def get(self, url, *, headers, timeout):
        self.gets.append((url, headers, timeout))
        if url.endswith("/root/delta"):
            return FakeResponse(
                payload={
                    "value": [
                        {
                            "id": "one",
                            "name": "one.txt",
                            "eTag": "v1",
                            "file": {},
                            "sensitivityLabel": {"displayName": "CUI"},
                            "permissions": [{"grantedToV2": {"group": {"id": "analysts"}}}],
                        },
                        {"id": "folder", "name": "ignored", "folder": {}},
                    ],
                    "@odata.nextLink": "https://graph.microsoft.us/v1.0/delta-page-2",
                }
            )
        if url.endswith("/delta-page-2"):
            return FakeResponse(
                payload={
                    "value": [
                        {"id": "gone", "name": "gone.txt", "deleted": {}, "eTag": "v2"},
                        {"id": "two", "name": "two.txt", "eTag": "v3", "file": {}},
                    ],
                    "@odata.deltaLink": "https://graph.microsoft.us/v1.0/final-delta",
                }
            )
        if url.endswith("/items/one/content"):
            return FakeResponse(content=b"first graph document")
        if url.endswith("/items/two/content"):
            return FakeResponse(content=b"second graph document")
        if url.endswith("/items/two/permissions"):
            return FakeResponse(payload={"value": []})
        raise AssertionError(f"unexpected Graph request: {url}")


class FakeTable:
    def __init__(self):
        self.items = {}

    def get_item(self, *, Key, ConsistentRead):
        item = self.items.get(next(iter(Key.values())))
        return {"Item": item} if item else {}

    def put_item(self, *, Item):
        key = Item.get("document_id") or Item.get("change_id")
        self.items[key] = Item

    def scan(self, **kwargs):
        items = [item for item in self.items.values() if not item.get("deleted")]
        if kwargs.get("Select") == "COUNT":
            return {"Count": len(items)}
        return {"Items": items}


class FakeS3:
    def __init__(self):
        self.puts = []
        self.deletes = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)


class FakeSearch:
    def __init__(self):
        self.requests = []

    def request(self, method, path, body="", content_type=""):
        self.requests.append((method, path, body, content_type))
        return {"errors": False, "items": [], "took": 1}


class IngestionTests(unittest.TestCase):
    def test_connector_protocol_and_fixture_count(self):
        connector = FilesystemConnector(FIXTURES, corpus="synthetic")
        self.assertIsInstance(connector, Connector)
        batch = connector.changes()
        self.assertGreaterEqual(len(batch.events), FIXTURE_DOCUMENT_COUNT)
        self.assertEqual({event.kind for event in batch.events}, {ChangeKind.UPSERT})
        self.assertEqual(
            [event.document.document_id for event in batch.events[:2]],
            ["enterprise-brief-001.txt", "enterprise-brief-002.txt"],
        )

    def test_deterministic_fixture_generation_and_cli_auto_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            first = generate_fixture_corpus(directory)
            first_contents = [path.read_bytes() for path in first]
            second = generate_fixture_corpus(directory)
            self.assertEqual(first_contents, [path.read_bytes() for path in second])
            self.assertEqual(len(first), 120)

            output = io.StringIO()
            with redirect_stdout(output):
                status = cli_main(["--source", directory])
            summary = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(summary["documents_generated"], 120)
            self.assertEqual(summary["documents_ingested"], 120)
            self.assertNotIn("fictional, unclassified data", output.getvalue())

    def test_chunking_and_normalized_security_metadata_are_deterministic(self):
        document = SourceDocument(
            corpus="enterprise",
            document_id="brief.txt",
            version="7",
            text="alpha beta gamma delta " * 20,
            security_label="cui",
            acl_principals=("group:z", "group:a", "group:a"),
            provenance={"path": "brief.txt"},
        )
        first = normalize_document(document, max_chars=64)
        second = normalize_document(document, max_chars=64)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all(len(record.text) <= 64 for record in first))
        self.assertEqual(first[0].security_label, "cui")
        self.assertEqual(first[0].acl_principals, ("group:a", "group:z"))
        self.assertEqual(first[0].corpus, "enterprise")
        self.assertEqual(first[0].document_id, "brief.txt")
        self.assertEqual(len(first[0].checksum), 64)
        self.assertEqual(len(first[0].document_checksum), 64)
        expected_chunks = chunk_text(document.text, max_chars=64)
        self.assertEqual(tuple(record.text for record in first), expected_chunks)

    def test_filesystem_idempotency_updates_and_deletes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "shared.txt"
            path.write_text("initial content", encoding="utf-8")
            connector = FilesystemConnector(
                root,
                corpus="shared",
                security_label="CUI",
                acl_principals=("group:j2",),
            )
            store = InMemoryIngestionStore(max_chunk_chars=64)

            first = ingest(connector, store)
            second = ingest(connector, store, cursor=first.cursor)
            self.assertEqual(first.applied.inserted, 1)
            self.assertEqual(second.changes, 0)
            original_record_ids = set(store.records)

            path.write_text("updated content with a different checksum", encoding="utf-8")
            third = ingest(connector, store, cursor=second.cursor)
            self.assertEqual(third.applied.updated, 1)
            self.assertNotEqual(original_record_ids, set(store.records))
            record = store.records_for("shared", "shared.txt")[0]
            self.assertEqual(record.acl_principals, ("group:j2",))

            path.unlink()
            fourth = ingest(connector, store, cursor=third.cursor)
            self.assertEqual(fourth.applied.deleted, 1)
            self.assertFalse(store.records)
            self.assertIn(("shared", "shared.txt"), store.tombstones)

            repeated = store.apply(fourth.applied and connector.changes(fourth.cursor).events)
            self.assertEqual(repeated.deleted, 0)

    def test_store_tombstone_event_and_identical_upsert_are_idempotent(self):
        document = SourceDocument("c", "d", "1", "safe source", acl_principals=("user:1",))
        store = InMemoryIngestionStore()
        event = ChangeEvent.upsert(document)
        self.assertEqual(store.apply([event]).inserted, 1)
        self.assertEqual(store.apply([event]).unchanged, 1)
        self.assertEqual(store.apply([ChangeEvent.delete("c", "d", version="2")]).deleted, 1)
        self.assertEqual(store.apply([ChangeEvent.delete("c", "d", version="2")]).unchanged, 1)

    def test_aws_store_persists_manifest_source_and_acl_index_records(self):
        documents = FakeTable()
        changes = FakeTable()
        s3 = FakeS3()
        search = FakeSearch()
        store = AwsIngestionStore(
            document_table=documents,
            change_table=changes,
            s3_client=s3,
            source_bucket="source-bucket",
            kms_key_arn="kms-key",
            search_client=search,
            search_index="enterprise-index",
            region="us-gov-west-1",
            security_domain="demo",
            embedder=lambda _text: [0.0] * 1024,
        )
        document = SourceDocument(
            corpus="shared",
            document_id="brief.txt",
            version="1",
            text="approved synthetic source",
            security_label="demo",
            acl_principals=("group:analysts",),
        )

        first = store.apply([ChangeEvent.upsert(document)])
        second = store.apply([ChangeEvent.upsert(document)])

        self.assertEqual(first.inserted, 1)
        self.assertEqual(second.unchanged, 1)
        self.assertEqual(store.document_count(), 1)
        self.assertEqual(len(s3.puts), 1)
        self.assertEqual(s3.puts[0]["ServerSideEncryption"], "aws:kms")
        bulk_body = next(
            body
            for method, path, body, _content_type in search.requests
            if method == "POST" and path.startswith("_bulk")
        )
        self.assertIn('"acl_principals": ["group:analysts"]', bulk_body)
        self.assertNotIn("approved synthetic source", json.dumps(documents.items))

    def test_graph_sovereign_delta_pagination_content_acl_and_tombstone(self):
        session = FakeGraphSession()
        client = GraphClient(
            client_id="external-client",
            client_secret="external-secret",
            tenant_id="tenant",
            graph_base_url="https://graph.microsoft.us/v1.0",
            token_url="https://login.microsoftonline.us/tenant/oauth2/v2.0/token",
            session=session,
        )
        connector = GraphDeltaConnector(
            client,
            drive_id="drive",
            corpus="graph",
            default_acl_principals=("tenant:default",),
        )
        batch = connector.changes()

        self.assertEqual(batch.cursor, "https://graph.microsoft.us/v1.0/final-delta")
        self.assertEqual(len(batch.events), 3)
        self.assertEqual(
            [event.kind for event in batch.events],
            [
                ChangeKind.UPSERT,
                ChangeKind.DELETE,
                ChangeKind.UPSERT,
            ],
        )
        self.assertEqual(batch.events[0].document.security_label, "cui")
        self.assertEqual(batch.events[0].document.acl_principals, ("group:analysts",))
        self.assertEqual(batch.events[2].document.acl_principals, ("tenant:default",))
        self.assertEqual(
            session.posts[0][0], "https://login.microsoftonline.us/tenant/oauth2/v2.0/token"
        )
        self.assertEqual(
            session.posts[0][1]["scope"],
            "https://graph.microsoft.us/.default",
        )
        self.assertTrue(
            all(call[1] == {"Authorization": "Bearer test-token"} for call in session.gets)
        )
        self.assertEqual(len(session.posts), 1)

    def test_graph_incremental_cursor_is_used_directly(self):
        class CursorClient:
            def __init__(self):
                self.urls = []

            def get_json(self, url):
                self.urls.append(url)
                return {"value": [], "@odata.deltaLink": "next"}

        client = CursorClient()
        batch = GraphDeltaConnector(client, drive_id="d", corpus="c").changes("saved-cursor")
        self.assertEqual(client.urls, ["saved-cursor"])
        self.assertEqual(batch.cursor, "next")


if __name__ == "__main__":
    unittest.main()
