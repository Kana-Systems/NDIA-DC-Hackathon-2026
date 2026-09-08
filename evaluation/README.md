# Offline evaluation

The benchmark uses fictional contracts in compliant, missing-clause, malformed,
DoD solely-COTS, and civilian-agency scenarios. No fixture is an official
procurement document or suitable for legal use.

Run from the repository root:

`python -m evaluation.benchmark`

Use `--output evaluation/report.json` to retain a report (generated reports
should not be committed). The runner loads `expected_outputs.json`, executes the
deterministic policy checks, and reports micro precision/recall/F1 plus exact
case match. A non-matching case produces exit status 1, making the command
suitable for CI. It also includes the local heuristic classifier output to
expose the no-model baseline behavior.

The expected-output contract contains fixture-relative paths and ordered
`expected_finding_ids` plus acquisition metadata. DFARS 204.7304(c) evaluation
requires `agency_department` and `instrument_type`, and excludes records with
`acquisition_solely_cots: true`; contract wording and dollar value are not used
to infer applicability. Full findings preserve rule IDs, severity, evidence,
and stable source citations. Update expected output only after deliberate policy
review. The sample runner mirrors the curated rule IDs in
`knowledge/rules/review_rules.yaml` and has no third-party dependencies.

## J2 intelligence evaluation

Run `python -m evaluation.intelligence_benchmark` to exercise the generated
120-document corpus and report retrieval Recall@k, citation correctness, ACL
leakage, ingestion idempotency, entity/change behavior, and target-object
approval enforcement. These metrics are separate from legal accuracy and model
quality. The fixture corpus is synthetic and is not evidence of access to an
operational SharePoint tenant or authoritative intelligence store.

An ACL leakage count above zero, an unresolved citation accepted as grounded,
or export of an unapproved object is a hard benchmark failure. Retrieval and
entity-resolution scores are regression indicators for the curated fixtures,
not production acceptance thresholds.

The checked deterministic baseline is 120 fixture documents and records,
Recall@5 of `1.0`, citation correctness of `1.0`, zero ACL leakage, zero changes
on the idempotent second ingestion, one expected entity change event, blocked
unapproved export, and successful approved JSON export. CI runs this benchmark
independently of the legal-review benchmark.
