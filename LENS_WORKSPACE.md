# Connected Acquisition Lens

`/lens/` is the only interface; `/` redirects there and `/ui/` returns 404.
The six pages share persistent server-side records instead of separate demos.

| Page | What works | Use case |
| --- | --- | --- |
| Contracts | Upload/paste, metadata, trained classifier + Bedrock review, findings, citations, contextual questions, history and human approval | Contract review |
| Ask / Research | Grounded answers, summaries and memo drafts; saved citation trails; create reviewable records | RAG Intelligence Service |
| Data connections | Approved shared folders or configured Graph connector; batch sync, changes, removals, errors and audit trail | Foundational Data Ingestion |
| Source library | Indexed workspace references and official FAR/DFARS corpus; discovery catalog clearly separate | Contract review + RAG |
| Entities & relationships | Analyst-entered exact-source excerpts; normalized-name merging; relationships; approval reset after changes | Foundational structured knowledge |
| Reviewed records | Cited drafts and reviews; notes; approval before JSON export; stale-source gates | Analyst-reviewed structured records |

## AWS runtime

The main-branch deployment builds the LFS-backed model/corpus image, deploys
ECS Fargate, and checks target health. Bedrock and the trained classifier remain
enabled. SageMaker infrastructure is retained; this change launches no training
jobs and does not replace the selected model.

`WORKSPACE_TABLE` selects the new on-demand DynamoDB table, with the existing
customer-managed J2 KMS key, point-in-time recovery, and deletion protection.
The task role can access it; credentials remain in Secrets Manager and the
task-role chain. Records are partitioned by security domain and user subject,
with conditional writes rejecting conflicting edits. Audit records exclude
document bodies and credentials. Source originals and parsed bodies are stored
in KMS-encrypted S3; DynamoDB holds record metadata, revisions, approvals and
audit events, not the full document bodies. Owner/domain-filtered OpenSearch
provides keyword + Titan-vector retrieval for private documents. An isolated,
snapshot-versioned OpenSearch index searches the official corpus lexically;
the packaged SQLite snapshot verifies the exact canonical citation text.
Uploaded documents and workspace databases
are never committed to Git/LFS. Synthetic deployment seed steps are disabled;
older fixtures are not deleted, but Lens does not retrieve from them.

Local development uses `WORKSPACE_DB_PATH` (SQLite). Docker Compose mounts a
named persistent workspace volume. Do not use `docker compose down -v` if the
workspace data should be retained.

## Demo boundaries

- Everyone using the shared demo password shares one identity and workspace.
  This is not production per-person authentication. The existing backend OIDC
  adapter requires an approved IdP and frontend login integration for production.
- Shared-folder sync needs `WORKSPACE_IMPORT_ROOT` pointing to an approved
  directory mounted on the server; ECS cannot access a laptop folder directly.
  The Lens SharePoint connector handles TXT, MD, PDF and DOCX with recursive
  pagination, bounded downloads, original-file retention, version tracking,
  deletion detection, and explicit errors. It does not follow remote shortcuts.
- SharePoint uses application `Sites.Selected` plus an explicit site read grant.
  An administrator must additionally approve a public/unclassified library for
  named workspace identities in the secret configuration. This is NOT live
  per-user SharePoint ACL synchronization. Failed sync or a sync older than one
  hour blocks its documents from retrieval/approval/export until access recovers.
- Sync can be triggered manually and runs automatically every 15 minutes in AWS
  when the connector secret ARN is configured. Conditional leases prevent
  overlapping syncs and allow recovery after a one-hour abandoned lease.
  The worker is in-process, not a durable distributed job queue.
- Tests process 101 temporary documents to exercise scale, updates and errors.
  This does not demonstrate ingesting 100 real SharePoint documents; approved
  original documents must be supplied to demonstrate that requirement.
- Entity entry and normalized-name matching are analyst-assisted, not semantic
  AI entity resolution. Structured exports are reviewed JSON, not integration
  with an authoritative target-system data store.
- Local/offline retrieval stays lexical. AWS indexing failures are visible and
  retryable; unavailable or superseded documents are excluded from retrieval.
  Existing pre-integration uploads need reindexing; they are not automatically
  migrated by deployment. New uploads and synced documents use S3/OpenSearch.
- AWS source ingestion supports the configured extracted-text limit (default
  2 million characters) in S3 with bounded indexing batches; local inline storage
  and full-contract model review still accept at most 30,000 characters. Metadata
  records are limited to 350 KB. Source changes/removals invalidate exports. Citation
  checks validate references, not legal entailment or compliance.
- Bedrock inference is billable. Opening the app starts no SageMaker training.

## SharePoint setup

Run `python scripts/configure-sharepoint.py` locally. Enter the secret VALUE only
in its hidden terminal prompt, never in chat, Git or a command-line argument.
Choose the library and explicitly approve sharing with the judge demo identity.
The script verifies the actual Graph site/library and writes the encrypted AWS
secret. Set the repository deployment secret `LENS_GRAPH_SECRET_ARN` to the ARN
it prints, then deploy main. The task role gets access to that exact secret.
In Data connections, test access, create a SharePoint connection with a relative
folder and correct category, then sync. The runtime connector remains read-only.

Before claiming production readiness: integrate individual OIDC logins and
source ACL changes, durable queued ingestion, long-document/OCR support,
semantic entity resolution, and held-out answer-quality evaluation. A real
100-document ingestion run meets that ingestion requirement only; it does not
prove every enterprise RAG or J2 requirement is complete.

## Validation

Run `pytest`, `ruff check docker-entrypoint.py app evaluation ingestion knowledge
ml tests`, `npm --prefix frontend test`, `npm --prefix frontend run lint`, and
`npm --prefix frontend run build`. Validate Terraform without applying using
`terraform init -backend=false` and `terraform validate` in `infra/terraform`.
Tests cover isolation, persistence, sync, contextual retrieval, conflicts,
approval/export gates, generation errors, safe rendering, and connected UI.
