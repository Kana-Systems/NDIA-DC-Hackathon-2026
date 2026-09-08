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
document bodies and credentials. Uploaded documents and workspace databases
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
  Graph needs administrator-supplied credentials and permissions. Its existing
  adapter ingests text; PDF/DOCX parsing is available via upload/shared folders,
  not Graph. No connected Microsoft 365 tenant is fabricated.
- Lens batch sync is user-triggered and runs in the background. Recurring Lens
  scheduling is not enabled. The separate legacy Graph scheduler does not
  populate the new Lens owner-scoped records.
- Tests process 101 temporary documents to exercise scale, updates and errors.
  This does not demonstrate ingesting 100 real SharePoint documents; approved
  original documents must be supplied to demonstrate that requirement.
- Entity entry and normalized-name matching are analyst-assisted, not semantic
  AI entity resolution. Structured exports are reviewed JSON, not integration
  with an authoritative target-system data store.
- Private workspace retrieval is lexical; the federal corpus uses SQLite FTS.
  Existing OpenSearch/vector infrastructure is retained but does not index the
  new private workspace records yet.
- Input sections are limited to 30,000 extracted characters; persisted records
  to 350 KB. Source changes/removals invalidate dependent exports. Citation
  checks validate references, not legal entailment or compliance.
- Bedrock inference is billable. Opening the app starts no SageMaker training.

## Validation

Run `pytest`, `ruff check docker-entrypoint.py app evaluation ingestion knowledge
ml tests`, `npm --prefix frontend test`, `npm --prefix frontend run lint`, and
`npm --prefix frontend run build`. Validate Terraform without applying using
`terraform init -backend=false` and `terraform validate` in `infra/terraform`.
Tests cover isolation, persistence, sync, contextual retrieval, conflicts,
approval/export gates, generation errors, safe rendering, and connected UI.
