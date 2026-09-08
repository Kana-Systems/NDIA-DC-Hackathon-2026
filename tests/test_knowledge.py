import hashlib
import io
import json
import sys
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from knowledge.index_opensearch import (
    DEFAULT_DIMENSIONS,
    OpenSearchRequestError,
    SigV4OpenSearchClient,
    index_records,
    prepare_records,
    resolve_service,
)
from knowledge.index_opensearch import main as index_main
from knowledge.ingest_dita import ingest_path
from knowledge.ingest_smart_matrix import ingest
from knowledge.opensearch_bulk import build_bulk, index_definition
from knowledge.schema import SourceRecord, make_citation_id, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "knowledge"


class KnowledgeTests(unittest.TestCase):
    def test_citation_id_is_stable_across_content_changes(self):
        first = SourceRecord("test", "FAR", "FAR", "FAR", "1.001", "One", "first").normalized()
        second = SourceRecord("test", "FAR", "FAR", "FAR", "1.001", "Two", "second").normalized()
        self.assertEqual(first.citation_id, second.citation_id)
        self.assertNotEqual(first.content_sha256, second.content_sha256)
        self.assertEqual(first.citation_id, make_citation_id("FAR", "FAR", "1.001"))

    def test_dita_ingestion(self):
        records = ingest_path(
            FIXTURES / "sample.dita",
            authority="FAR",
            document_id="FAR-test",
            document_title="Synthetic FAR",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].locator, "52.999-1-a")
        self.assertIn("deterministic test evidence", records[0].text)

    def test_smart_matrix_csv_ingestion(self):
        records = ingest(FIXTURES / "smart_matrix.csv")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].locator, "252.999-7000")
        self.assertIn("Applicability:", records[0].text)

    def test_sample_records_and_opensearch_payload(self):
        records = read_jsonl(str(ROOT / "knowledge" / "samples" / "official_sources.jsonl"))
        self.assertEqual(len(records), 3)
        policy = next(record for record in records if record["locator"] == "204.7304(c)")
        self.assertEqual(policy["citation_id"], "dfars:dfars:204-7304-c:5343579076a3")
        self.assertEqual(
            policy["content_sha256"],
            "c6b9d6e7427bea9cd27133734fab8325e1648d189a04bacefc718f04803af7c5",
        )
        self.assertEqual(policy["retrieved_at"], "2026-09-04")
        self.assertEqual(policy["effective_date"], "2026-05-07")
        self.assertTrue(policy["url"].endswith("contract-clauses."))
        self.assertEqual(
            hashlib.sha256(policy["text"].encode()).hexdigest(),
            policy["content_sha256"],
        )
        mapping = index_definition(DEFAULT_DIMENSIONS)
        self.assertEqual(
            mapping["mappings"]["properties"]["embedding"]["dimension"],
            1024,
        )
        payload = build_bulk(records, "test-index")
        lines = payload.splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(json.loads(lines[0])["index"]["_id"], records[0]["citation_id"])

    def test_embedding_preparation_validates_dimensions(self):
        records = [{"citation_id": "test:1", "text": "sample"}]
        prepared = prepare_records(records, lambda _text: [0.0] * 1024)
        self.assertEqual(len(prepared[0]["embedding"]), 1024)
        self.assertNotIn("embedding", records[0])
        with self.assertRaises(ValueError):
            prepare_records(records, lambda _text: [0.0] * 3)

    def _fake_client(self, *, exists, bulk_errors=False):
        class FakeClient:
            def __init__(self):
                self.calls = []
                self.exists = exists

            def request(self, method, path, body="", content_type=""):
                self.calls.append((method, path, body, content_type))
                if method == "HEAD":
                    if not self.exists:
                        raise OpenSearchRequestError(method, path, 404)
                    return {}
                if method == "PUT":
                    self.exists = True
                    return {"acknowledged": True}
                return {
                    "errors": bulk_errors,
                    "items": [{"index": {"error": {"type": "failure"}}}] if bulk_errors else [],
                    "took": 4,
                }

        return FakeClient()

    def test_absent_index_is_created_before_bulk_upsert(self):
        client = self._fake_client(exists=False)
        records = [{"citation_id": "test:1", "text": "sample"}]
        result = index_records(client, records, "test-index", 1024)
        self.assertEqual(result["took"], 4)
        self.assertEqual([call[0] for call in client.calls], ["HEAD", "PUT", "POST"])

    def test_existing_index_skips_creation(self):
        records = [{"citation_id": "test:1", "text": "sample"}]
        client = self._fake_client(exists=True)
        index_records(client, records, "test-index", 1024)
        self.assertEqual([call[0] for call in client.calls], ["HEAD", "POST"])

        skip_client = self._fake_client(exists=False)
        index_records(skip_client, records, "test-index", 1024, create_index=False)
        self.assertEqual([call[0] for call in skip_client.calls], ["POST"])

    def test_bulk_errors_raise_sanitized_failure_count(self):
        client = self._fake_client(exists=True, bulk_errors=True)
        records = [{"citation_id": "test:1", "text": "sample"}]
        with self.assertRaisesRegex(RuntimeError, "reported 1 failures"):
            index_records(client, records, "test-index", 1024)

    def test_sigv4_transport_handles_empty_and_sanitizes_404(self):
        class FakeCredentials:
            def get_frozen_credentials(self):
                return "credential-secret"

        class FakeRequest:
            def __init__(self, **kwargs):
                self.headers = kwargs["headers"]

            def prepare(self):
                return self

        class FakeSigner:
            def __init__(self, *_args):
                pass

            def add_auth(self, request):
                request.headers["Authorization"] = "credential-secret"

        class EmptyResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b""

        client = object.__new__(SigV4OpenSearchClient)
        client.endpoint = "https://search.example"
        client.region = "us-gov-west-1"
        client.service = "es"
        client.credentials = FakeCredentials()
        client._request_type = FakeRequest
        client._signer_type = FakeSigner

        with patch("urllib.request.urlopen", return_value=EmptyResponse()):
            self.assertEqual(client.request("HEAD", "test-index"), {})

        error = urllib.error.HTTPError(
            "https://search.example/test-index",
            404,
            "Not Found",
            {},
            io.BytesIO(b"credential-secret and response details"),
        )
        with (
            patch("urllib.request.urlopen", side_effect=error),
            self.assertRaises(OpenSearchRequestError) as raised,
        ):
            client.request("HEAD", "test-index")
        self.assertEqual(raised.exception.status_code, 404)
        self.assertNotIn("credential-secret", str(raised.exception))

    def test_sigv4_service_resolution(self):
        self.assertEqual(resolve_service("https://abc.aoss.us-gov-west-1.amazonaws.com"), "aoss")
        self.assertEqual(resolve_service("https://search.example"), "es")

    def test_dry_run_with_embeddings_makes_no_aws_calls(self):
        source = ROOT / "knowledge" / "samples" / "official_sources.jsonl"
        arguments = ["index_opensearch", str(source), "--dry-run", "--embed"]
        output = io.StringIO()
        with patch.object(sys, "argv", arguments), redirect_stdout(output):
            index_main()
        summary = json.loads(output.getvalue())
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["embedding_dimensions"], 1024)
        self.assertEqual(summary["embedding_requests"], 0)


if __name__ == "__main__":
    unittest.main()
