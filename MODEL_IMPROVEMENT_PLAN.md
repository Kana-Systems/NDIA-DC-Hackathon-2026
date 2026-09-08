# Model improvement and downstream benefit

## Scope and safeguards

Preserve `artifacts/models/legal-bert-cuad` and the existing selected registry.
Do not deploy AWS infrastructure. The user approved one live benchmark with a
$10 conservative estimated-spend cutoff: six cases, three arms, two repeats,
at most 96 API attempts including grading. `evaluation.cost_guard` persists
reservations before calls; failed/unknown-usage calls retain their reservations.
This is not an account-wide billing cap and excludes taxes/other resources.
Model quality improvements are hypotheses, not guaranteed outcomes. No automatic
promotion based on CUAD scores or synthetic/LLM-graded legal cases.

## Work packages

1. Cache unthresholded predictions. Tune a global threshold, then supported
   per-label thresholds on a document-disjoint calibration subset of validation.
   Compare on the remaining validation documents. Report precision, recall, F1,
   F2, macro scores and per-label support. Test is reporting-only; because the old
   test has already been inspected, call repeat results regression measurements,
   not a new untouched confirmatory evaluation.
2. Add bounded positive-class weighting, deterministic seeds, training progress,
   provenance and resume support. Spread negative examples across each training
   contract instead of selecting only its beginning. Train separate candidates.
3. Create source-pinned federal synthetic contrast cases, explicit applicability
   assumptions, review rubrics, train/eval separation, and human-label import
   validation. Synthetic labels are drafts and are NOT federal ground truth.
4. Compare LLM-only, LLM+retrieval, and LLM+retrieval+classifier with identical
   documents, repeated paired cases, blinded review packets, citation checks,
   failures, latency, and actual token usage. Machine grading is provisional.
5. Benchmark a second pretrained encoder against Legal-BERT and the trained
   linear baseline. Export a candidate registry, never overwrite the incumbent.

## Long-term release gate

Require reviewed federal cases, positive paired downstream improvement without
increased unsupported claims/missed critical issues, and acceptable cost/latency.
Record model, prompt, corpus and benchmark versions. Re-run on relevant changes;
reject stale evidence and retain the classifier-off fallback. Human reviewers
must adjudicate applicability and semantic citation support. No result from this
small initial benchmark establishes future or production legal reliability.

## Implementation status (2026-09-08)

- Upstream `origin/main` fast-forwarded to `8c0a223`; local edits restored cleanly.
  Recovery stash `recovery: local acquisition-lens before upstream 8c0a223 merge`
  retained. No push or new local code commit.
- Initial calibration/selection experiment completed. Neither per-label tuned
  candidate beat the incumbent under selection micro-F2 with precision >= .75.
  The fixed-threshold Legal-BERT remains selected. These are exploratory results:
  the incumbent already used validation for checkpoint selection.
- Weighted Legal-BERT continuation (one additional epoch, cap 4, learning rate
  1e-5) is running. RoBERTa base (three epochs, cap 4) runs next on the same
  document split. These have unequal cumulative training budgets; comparison is
  a practical candidate selection, not an architecture-controlled experiment.
- Fixed DITA extraction of main clause paragraphs that precede Alternate
  sections. Rebuilt separate `artifacts/knowledge/federal-v2.sqlite`: 5,497
  documents, 12,574 passages (640 recovered), 17,920 references. Original corpus
  retained. This is the same source revision, not a newer legal effective date.
- Six contrast cases and their rubrics are synthetic drafts. No attorney-reviewed
  federal labels or organizational playbook were supplied. `ml.federal_data`
  validates reviewed input provenance and rejects synthetic/draft labels and
  document/template-family leakage. It cannot authenticate reviewer credentials.
- `scripts/finish-model-improvements.py` waits without LLM polling, runs six
  complete CUAD contracts using annotation-independent sliding windows, then
  the approved paid ablation. No candidate is automatically deployed.

## Commands and outputs

```bash
# Local training (do not start another copy while the current job is running)
.venv/bin/python scripts/improve-models.py
# Follow-up: approved paid evaluation, max $10 estimated spend
.venv/bin/python scripts/finish-model-improvements.py
# Prepare benchmark without any paid calls
.venv/bin/python -m evaluation.model_benefit --candidate artifacts/improvement/initial/candidate.json --output artifacts/improvement/preflight-new
```

Read `artifacts/improvement/training-status.json` and checkpoint `progress.json`
for training. Follow-up stages are in `artifacts/improvement/final-checks-status.json`.
Results: `final/comparison.json`, `full-contract/report.json`,
`llm-benefit/summary.json`, and `llm-benefit/cost-ledger.json` below that directory.
Human packets intentionally omit variant/model and machine grades. Copy them
to a separate reviewed directory and fill reviewer identity, timestamp, independent
human review flag, and grade. The synthetic initial cases will still fail the
production release gate even after review.

```bash
.venv/bin/python -m evaluation.benefit_gate artifacts/improvement/llm-benefit --reviewed-packets PATH_TO_REVIEWED_PACKETS
```

The gate checks version fingerprints, complete independent human review,
non-synthetic case count/families, positive paired improvement, no increase in
critical misses/unsupported claims, and explicit latency/token budgets. It emits
a recommendation only; promotion still requires owner review. Thresholds are
decision cutoffs, not calibrated legal confidence. Model scores must never be
presented as compliance probability.
