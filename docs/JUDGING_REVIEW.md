# Hackathon review - September 9, 2026

Starting point: `bbc5da0`, the 9:41 AM update on `origin/main`. Improvements are
on `improve/judging-workflows` in the main hackathon checkout. The morning model
selection, SageMaker support, and deployment configuration are preserved.

## Rubric coverage

This maps the supplied rubric to reviewable behavior; it is not a claimed judge
score or a model-quality assessment.

| Criterion | Weight | Demonstrable behavior | Remaining evidence needed |
| --- | --- | --- | --- |
| Mission impact | 30% | Contracts → priority triage → source passages → human decision. Research starters produce obligation summaries, evidence-gap questions, and decision memo prompts. Reviewed records exposes pending, blocked, and exportable outputs. | User study with qualified reviewers; measured time saved on representative mission tasks. |
| Technical innovation | 25% | Existing classifier and cited generation pipeline is retained. Source/context versions follow research into reusable records; one readiness evaluator serves the UI and write gates. | Held-out model comparison and live endpoint run with recorded model provenance. No accuracy gain is claimed for UI changes. |
| Usability & design | 20% | Highest-priority findings first; combined search and evidence filters; honest source status; keyboard-operated contract tabs; visible progress; field validation; session renewal preserves drafts. | Visual browser review at desktop and mobile sizes; moderated task completion tests. |
| Security & sustainability | 15% | Source and metadata changes block approval/export; unresolved citations cannot pass on a label alone; stale reviews can be returned for revision; existing owner/domain isolation remains. | Individual OIDC login, source ACL synchronization, durable ingestion jobs, and operational approval remain production work. |
| Team collaboration | 10% | Decision notes show actor/time. Contract records link back to original context. This walkthrough and a versioned handoff specification make the demo and integration repeatable. | A rehearsal with named presenter, reviewer, and integration roles; independent reviewer feedback. |
| Interoperability bonus | Screenshot: 1%; badge: 0.05x | Downloadable JSON Schema, source/version manifest, checksum, and an offline validator support another team's consumer. | A real authorized dataset exchange and successful independent consumer import. The screenshot's two bonus values require organizer clarification before calculating a total. |

## Six-minute demonstration

1. **Mission question (30 seconds):** explain who must make the acquisition or
   analyst decision and what evidence they need. Use approved public/synthetic
   input. Establish the active generation mode from the actual output.
2. **Source (60 seconds):** add a contract and an attributed reference, or sync
   an approved source connection. Inspect source readiness and acquisition
   context. Show that discovery catalog links are separate from ingested text.
3. **Contract review (90 seconds):** run a review. Open High priority, then Needs
   evidence. Inspect one contract excerpt, recommendation, and supporting source.
   Use Ask about this finding to retain the exact context.
4. **Research (60 seconds):** choose Draft a decision memo, edit the prompt, and
   submit it. Inspect individual statement citations, then save a reviewable
   record. Prompt buttons do not invoke the model until submission.
5. **Human decision (60 seconds):** open Knowledge → Reviewed records. Inspect
   an output and its source readiness. Record what was checked before approval.
   Show attribution, then filter Ready to export.
6. **Handoff and failure behavior (60 seconds):** download JSON and its schema;
   validate the handoff. In an isolated demo workspace, change contract metadata
   and show the resulting blocker on the previous review. Return the stale
   result for revision. End with the next action the mission owner can take.

Allow more time when live generation is slow. Do not narrate the offline
extractive mode as a successful live model invocation.

## Architecture decisions

- **One readiness boundary:** `app/workspace_readiness.py` computes blockers
  from current, owner-scoped data. List responses show this result; approval and
  export recompute it. Display state never becomes a permission grant.
- **Context is provenance:** a question saves the selected document version at
  generation time. Creating a structured record copies that snapshot. A
  reference-only answer still becomes stale when its contract context changes.
- **Keep recoverable work:** rejection is allowed for stale outputs. Session
  renewal keeps React forms mounted and does not automatically replay writes.
  Late authentication errors cannot clear a newer token.
- **Bound handoffs:** the export adapter owns the schema, manifest, and checksum.
  It does not write to an external target. The offline validator checks structure
  and integrity, not legal correctness or ongoing access rights.
- **Reusable UI behavior:** finding triage is shared by the contract and record
  views. Contract tabs use manual keyboard activation following the
  [WAI-ARIA tabs pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/).

## Validation and limits

Completed local checks:

| Check | Result |
| --- | --- |
| Backend regression suite | 219 passed |
| Frontend regression suite | 26 passed |
| Python lint and formatting | Passed, including the handoff validator |
| Frontend lint and production build | Passed |
| Repository `scripts/check-local.sh` | Passed; the final export edge case was also checked afterward |
| Deterministic contract benchmark | 5/5 cases matched |
| Synthetic intelligence benchmark | Passed: 120 documents, zero ACL leakage, repeat ingestion produced no changes |
| Local HTTP workflow | Authentication, contract review, approval, schema/checksum validation, ingestion, research, changed-source export blocking, and stale return for revision passed |
| Published export schema | Matches the API model |

The repository formatter found pre-existing drift in seven evaluation/ML/test
files. Formatting was normalized and their Python syntax trees were confirmed
unchanged. No model selection, training logic, inference configuration, or
deployment behavior was changed.

Backend regression tests cover readiness, source/context changes, missing
citations, owner/domain isolation, and export integrity. Frontend tests cover
triage, keyboard navigation, stale decision controls, research prompts, and
session renewal without form loss or duplicate writes. The existing deterministic
contract and 120-document intelligence benchmarks remain separate from claims
about real-world model quality.

The connected Browser tool reported no available browser in this session, so
pixel-level layout and live browser interactions were not verified. Cloud
deployment and live billable inference were not part of these local checks.
