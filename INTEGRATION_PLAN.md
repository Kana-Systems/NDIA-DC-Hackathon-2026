# Acquisition Lens integration

## Objective

Use this repository as the canonical backend for the federal contract review
hackathon, with the Acquisition Lens React interface from the standalone local
prototype. Work on `integration/acquisition-lens`; preserve the original local
prototype. The expanded scope includes actual local classifier training and live
GPT-5.6 Terra inference through the existing GovCloud credentials. Cloud
infrastructure deployment remains out of scope.

## Decisions

| Concern | Keep / combine |
| --- | --- |
| Review engine | Repository parsing, metadata and report contracts; new model review uses learned categories, retrieved source text and Terra-generated findings. Preserve `ReviewService` as the explicit offline baseline. |
| Judge experience | Local prototype React layout, password access, sample loading, findings, source library, relationship view |
| Authentication | Repository demo-token issuance and validation; no second password/token implementation |
| Data sources | Local catalog of the 20 requested sources, clearly distinguished from evidence actually retrieved by the backend |
| ML | Existing CUAD preparation/training/evaluation tooling; surface actual model IDs and missing confidence, never manufacture an accuracy score |
| Knowledge / explanations | Versioned FAR/DFARS corpus with full-text retrieval and source graph; reuse the existing Terra Bedrock transport for query planning and contract review. Model mode is the default local launch. |
| Uploads / advanced workflows | React PDF/DOCX uploads reuse the repository parser; retain Gradio and J2 workflows |
| Infrastructure | Preserve repository Docker/Terraform; do not deploy or alter cloud resources in this pass |

## First implementation slice

1. Import the React application and source catalog without generated dependencies.
2. Add a thin browser API adapter that calls the existing review engine, validates
   acquisition metadata, and reuses repository identity. Keep the complete native
   report alongside the display mapping so provenance and clause status survive.
3. Collect acquisition metadata in React, load sample metadata, show actual engine
   modes, grounding status and clause inventory, and remove misleading confidence
   or ingestion claims. Keep synthetic sample and live results distinct.
4. Use a same-origin development proxy and serve a built React app at `/lens/`;
   keep existing `/ui/` and versioned API routes available.
5. Add local setup/run/check scripts, meaningful API/UI integration tests, and run
   the repository tests and offline benchmarks plus frontend tests/lint/build.

## Follow-on work

- Extend the ingested FAR/DFARS snapshots to Smart Matrix, agency-specific
  supplements, awards and adjudication sources as each connector is verified.
  Preserve effective-date metadata where published; do not infer that retrieval
  time is the law's effective date. Other catalog entries are not ingested yet.
- Add organization-provided approved playbooks and labeled federal-contract
  evaluation examples. CUAD categories are commercial topics, not compliance labels.
- Add asynchronous batching for documents above the current 30,000-character
  model review limit and evaluate cross-section reasoning on long documents.
- Deploy the integrated React bundle, trained artifact and indexed corpus through
  the repository infrastructure when a deployment is requested.

## Completed and active work

- React interface, same-origin API, shared identity, metadata collection, actual
  evidence relationships, and PDF/DOCX support are implemented.
- Live Terra access and combined authenticated model review have been verified.
- 5,497 FAR/DFARS DITA documents have been ingested into 11,934 passages and
  17,920 source-reference links. Source commits and content hashes are recorded.
- A multi-label logistic classifier was trained on actual CUAD annotations.
  The initial 395/115 contract split achieved micro-F1 0.5906, precision 0.8930,
  recall 0.4411. These are clause-window metrics, not legal accuracy.
- A second comparison uses 319 training, 76 validation and 115 test contracts.
  Legal-BERT fine-tuning completed on this split: validation micro-F1 0.6701
  versus linear 0.5539; selected Legal-BERT test micro-F1 0.6933, precision 0.8727,
  recall 0.5750. The live Terra review smoke check passed. These are commercial
  clause-window metrics, not federal compliance or end-to-end legal accuracy.
- Further experiments, upstream merge status, source extraction corrections and
  the $10-capped downstream comparison are tracked in `MODEL_IMPROVEMENT_PLAN.md`.
- Regression tests and both inherited offline benchmarks pass. Browser automation
  is unavailable in this session; UI verification uses component tests and a live
  HTTP workflow, not a claimed visual inspection.

## Acceptance

Successful authenticated sample and custom-text reviews must execute the native
engine. Invalid credentials, absent metadata and malformed input must fail.
Displayed citations must resolve to the native report; unsupported findings stay
unverified. No trained model or live ingestion is claimed merely because code for
it exists. The legacy repository tests and offline benchmarks must still pass.
