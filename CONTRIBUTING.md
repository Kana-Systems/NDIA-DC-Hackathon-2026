# Contributing to Kana Legal

Start with [local setup](docs/SETUP.md), then rehearse the
[connected use cases](docs/JUDGING_REVIEW.md). Use public or synthetic documents
in examples, tests, and screenshots. Keep downloaded inputs, local workspaces,
credentials, and generated review exports out of Git.
Never submit tenant identifiers, real contracts, operational intelligence, PII,
privileged material, CUI, classified information, or export-controlled content.
Confirm redistribution rights and record provenance for new source material.

## Where to make changes

| Area | Code | Responsibility |
| --- | --- | --- |
| Application and API | `app/main.py`, `app/api.py`, `app/workspace.py` | Startup, authenticated routes, bounded requests |
| Contract review | `app/model_review.py`, `app/service.py`, `ml/` | Live/model and deterministic review paths, classifier inference |
| Workspace persistence | `app/workspace_store.py` and `app/workspace_*.py` | Owner/domain scoping, versions, source sync, audit records |
| Review readiness | `app/workspace_readiness.py` | Recompute evidence and freshness blockers before approval/export |
| Retrieval and ingestion | `knowledge/`, `ingestion/` | Official corpus, source parsing, access-filtered retrieval |
| Interface | `frontend/src/` | React workflows, session handling, accessible controls |
| Hosting | `infra/terraform/`, `.github/workflows/deploy.yml` | GovCloud resources, immutable images, rollout and smoke checks |

Keep the server authoritative for access, freshness, and write validation.
Client readiness indicators explain server decisions; they do not authorize
writes. Preserve source/version snapshots through research, review, and export.
Live model failures must remain visible, with no silent offline substitution.
See [architecture decisions](docs/JUDGING_REVIEW.md#architecture-decisions) and
the [export contract](docs/INTEROPERABILITY.md) before changing those boundaries.

## Comments and documentation

Write docstrings for public helpers and comments for non-obvious decisions:
why a source check happens at write time, why an authentication retry must not
replay a write, or why a SageMaker resource has a particular storage setting.
Keep comments near the relevant code and update them with behavior changes.
Avoid comments that merely repeat a variable assignment or function name.

Examples already live beside the readiness checks, session handling, container
entrypoint, SageMaker configuration, and deployment permission reconciliation.
Document new environment variables in `.env.example`; update setup instructions
when changing dependencies, commands, ports, or launch defaults. Add export
schema/version guidance when changing data consumed by another team.

## Validate a change

```bash
bash scripts/check-local.sh
```

This runs the backend and frontend regression suites, lint, deterministic
benchmarks, and the frontend production build. Add focused regression coverage
for changed behavior or a reproduced bug. Use synthetic fixtures and mocked
external services; keep live cloud checks explicit and separate.

For infrastructure, also run the workflow's Terraform formatting and validation
commands (`terraform fmt -check -recursive` and `terraform validate` under
`infra/terraform`, after initialization). Inspect the plan before applying
resource changes. Deployment checks
include the pinned classifier artifact, ECR base-image access, SageMaker runtime
response, ECS stability, and load-balancer target health.

Explain the problem, resulting behavior, relevant checks, and remaining limits
in a change description. Update the README and linked guides alongside the code
so another teammate can set up and demonstrate the same revision.

Contributions intentionally submitted to this project are provided under the
Apache License 2.0 as described in `LICENSE`. Submitting a change represents
that you have the right to contribute it under those terms.
