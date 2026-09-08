# Contributing

Use only public or synthetic test data. Never submit credentials, tokens,
Terraform state, tenant identifiers, real contracts, operational intelligence,
PII, privileged material, CUI, classified information, or export-controlled
content.

Before proposing a change:

1. Run `python -m ruff format --check app evaluation ingestion knowledge ml tests`.
2. Run `python -m ruff check app evaluation ingestion knowledge ml tests`.
3. Run `python -m pytest -q`.
4. Run both deterministic evaluation modules documented in `evaluation/README.md`.
5. Run `terraform fmt -check -recursive` and `terraform validate` under
   `infra/terraform` for infrastructure changes.
6. Confirm new source material has redistribution rights and recorded
   provenance.

Contributions intentionally submitted to this project are provided under the
Apache License 2.0 as described in `LICENSE`. Submitting a change represents
that you have the right to contribute it under those terms.
