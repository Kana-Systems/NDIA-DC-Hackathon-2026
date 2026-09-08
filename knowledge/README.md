# Knowledge ingestion

This module converts official acquisition publications and Smart Matrix exports
to deterministic JSON Lines records. It is intentionally network-free: download
official artifacts separately, record their retrieval date, then ingest locally.

## Interfaces

- `python -m knowledge.ingest_dita SOURCE OUTPUT --authority FAR --document-id FAR --document-title "..."`
  accepts a DITA/XML file, directory, or ZIP.
- `python -m knowledge.ingest_smart_matrix MATRIX.csv OUTPUT` accepts CSV with
  no dependencies. XLSX/XLSM needs `openpyxl>=3.1`.
- `python -m knowledge.opensearch_bulk RECORDS OUTPUT --mapping-output mapping.json
  [--vector-dimensions 1024]` creates newline-delimited Bulk API input and an
  OpenSearch mapping. The `text` field uses BM25; optional `embedding` values
  use a Lucene cosine HNSW `knn_vector`.
- `python -m knowledge.index_opensearch RECORDS --dry-run --embed
  --bulk-output bulk.ndjson --mapping-output mapping.json` prepares the same
  1024-dimensional mapping and payload without credentials or network calls.

Run commands from the repository root. Inputs are never modified and no command
contacts OpenSearch unless `knowledge.index_opensearch` is run without
`--dry-run`.

## AWS OpenSearch indexing

For a live GovCloud indexing run:

`python -m knowledge.index_opensearch RECORDS --endpoint
https://search-DOMAIN.us-gov-west-1.es.amazonaws.com --region us-gov-west-1
--embed`

The command signs index-creation and Bulk API requests with the active boto3
credential chain. It automatically uses service name `aoss` for OpenSearch
Serverless endpoints and `es` otherwise; override with `--service` if needed.
Each deployment first sends a signed `HEAD` request. An absent index is created
with the requested BM25/vector mapping; an existing index is left unchanged.
Bulk actions use stable citation IDs and OpenSearch's `index` action, so later
runs upsert records without creating duplicates. A concurrent creator between
the `HEAD` and `PUT` is detected safely.

`--skip-create` targets an index managed elsewhere and bypasses both `HEAD` and
`PUT`; only the bulk upsert is sent. Transport failures expose only the request
method, index path, and HTTP status—not AWS credentials or response bodies.

With `--embed`, each record is enriched through Amazon Titan Text Embeddings V2
(`amazon.titan-embed-text-v2:0`) using normalized 1024-dimensional vectors.
Use `--bedrock-region` if the enabled GovCloud Bedrock region differs from the
OpenSearch region. The AWS identity needs `bedrock:InvokeModel` plus applicable
OpenSearch HTTP/index permissions. Model access and availability must be enabled
in the selected account and region.

Dry runs deliberately make zero embedding requests and add no placeholder
vectors; they validate records and generate an embedding-capable mapping. Run
without `--dry-run` to generate real vectors and index records.

## Record contract

`SourceRecord` requires `authority`, `document_id`, `locator`, and non-empty
`text`. Citation IDs derive only from those stable source coordinates; a text
change updates `content_sha256` without breaking stored citations. JSON output
uses sorted keys and normalized whitespace for reproducible diffs.

Enterprise records additionally carry `corpus_id`, `source_document_id`,
`version`, `security_label`, `acl_principals`, `entity_ids`,
`parent_citation_id`, `ingested_at`, and `deleted`. ACL principals must never be
empty. Query-time OpenSearch filters and an application trust-boundary check
both enforce those fields; a model-provided citation cannot bypass them.

`samples/official_sources.jsonl` allows offline demonstrations. Rules and the
review playbook are curated YAML data, not legal determinations. Source
attribution and canonical locations are in `manifests/sources.yaml` and
`ATTRIBUTION.md`.

## Optional dependencies

Core ingestion supports Python 3.9+ standard library only. Install
`openpyxl>=3.1,<4` solely for XLSX ingestion. Live AWS indexing uses the
project's `boto3`/`botocore` dependency. Treat downloaded official files as
versioned inputs and retain checksums in deployment-specific provenance logs.
