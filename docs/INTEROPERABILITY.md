# Workspace handoff contract

Reviewed records exports a version **1.1** JSON envelope. The original `record`
field is retained from 1.0. Version 1.1 adds a deduplicated `sources` manifest,
`record_sha256`, and an explicit `hash_format`. Consumers that require exactly
`schema_version == "1.0"` must opt into 1.1 before accepting these exports.

Download the schema using **Reviewed records → Export schema**, or call the
authenticated `GET /api/workspace/export-schema` endpoint. A matching copy is
included at [workspace-export.schema.json](workspace-export.schema.json).

The source manifest includes document ID, version, title, source type, security
label, citation IDs, and whether a document provided context, cited evidence,
or both. Citations and excerpts remain in the original record. The contract
does not grant permission to another team or write into another system.
Legacy citations with missing document IDs or versions retain blank values;
unrelated citations with missing IDs remain separate manifest entries.

## Validation by a receiving team

```bash
.venv/bin/python scripts/verify-workspace-handoff.py downloaded-record.json
```

This command validates the envelope, approval state, and checksum without making
network calls or logging document content. It exits nonzero for invalid input.
Use the published JSON Schema when implementing a consumer in another language.

`record_sha256` covers only the `record` object. To reproduce the canonical bytes
in Python:

```python
json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
```

Hash those bytes with SHA-256. The hash is stable across export timestamps and
detects accidental record changes. It is not a signature or a proof of origin;
it does not authenticate the source manifest or establish continued access.
Receivers must retain security labels and follow their approved sharing process.
Check current source access and freshness in the originating workspace before
reuse. A downloaded artifact cannot revoke itself after a source changes.

## Approval and freshness

The server recalculates readiness on every approval and export request. Review
and research records are blocked when relevant workspace sources are missing,
unavailable, or changed. Contract acquisition metadata is checked against the
review snapshot. Research records retain their context document version even
when the answer cites only other sources. Research statements must contain
resolvable citation IDs; a `verified` label alone is insufficient.

Context-bound research records created before this change have no context
version snapshot. Generate a fresh answer and record before approving or
exporting those records. Existing records are retained for inspection.

Readiness is computed per request and never stored as an authorization. The UI
shows the same blockers as the server, but a successful read does not authorize a
later export. Availability checks reuse source metadata within a request and do
not download document bodies from S3.
