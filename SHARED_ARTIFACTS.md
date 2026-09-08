# Shared model and regulatory database

The `first-test-merge` branch includes the completed Legal-BERT classifier and
the corrected FAR/DFARS SQLite corpus. Only weights and the database use Git LFS;
the tokenizer, model configuration, labels, provenance, metrics, and attribution
are ordinary Git files. Install Git LFS before cloning, then run `git lfs pull`.
LFS storage/downloads count against the repository owner's LFS allowance; they
are separate from AWS inference costs.

```bash
git lfs pull
git lfs fsck
python3 scripts/verify-shared-artifacts.py
bash scripts/setup-local.sh
bash scripts/run-local.sh
```

Open http://127.0.0.1:8080/lens/ (local demo password `contract-demo`). Setup
downloads dependencies, not training data, and does not retrain anything.
Actual LLM reviews still need your own authorized AWS GovCloud credentials and
Bedrock model access. Never commit or exchange credentials. Interactive reviews
are billable and are not covered by the separate evaluation's $10 cutoff.

## Included snapshot

- Model: `artifacts/models/legal-bert-cuad`, ID
  `legal-bert-cuad-79359150828d`, fine-tuned on CUAD's 41 clause labels.
- Weights: `model.safetensors`; no pickle training state is needed for inference.
- Corpus: `artifacts/knowledge/federal-v2.sqlite`, 5,497 documents, 12,574 chunks,
  and 17,920 reference links. The launcher uses this corrected snapshot by default.
- FAR source: https://github.com/GSA/GSA-Acquisition-FAR/tree/da52ccbbe114e1f031a7f4c59195c508dbfa485f
- DFARS source: https://github.com/GSA/GSA-Acquisition-DFARS/tree/7e609f791af9cc6d8e7d75a7b05c83b1f62c0cb8

The database contains ingested public regulatory text, its provenance, a
full-text index, and extracted reference edges, not uploaded contracts or user
accounts. It is a versioned local snapshot, not a synchronized multi-user server.
Source snapshot dates do not guarantee current legal applicability. Other source
catalog entries, including SAM and agency decisions, are not included.

The completed model's held-out CUAD clause-window micro-F1 is 0.6933, precision
0.8727, and recall 0.5750. These are not federal compliance accuracy or calibrated
legal confidence. Sharing this baseline does not promote an experimental model
or prove that classifier hints improve the LLM. Human legal review remains required.

Fine-tuned weights retain CC BY-SA 4.0, separately from the application's license.
See the model's `PRETRAINED_ATTRIBUTION.md` and `CUAD_ATTRIBUTION.md` for source
credit, modifications, and license links. Raw CUAD data is not redistributed here.

## Updating the shared artifacts

Keep the completed model's tokenizer, configuration, labels, metrics and
provenance together with its weights. Review candidate evaluations before
changing `selected.json`; do not commit a checkpoint that is still being written.
Keep credentials, uploads, raw training data, caches, optimizer state, and local
evaluation logs ignored. Commit corpus updates only after verifying their source
provenance and integrity. Re-run the verifier and `git lfs fsck` before pushing.
Git LFS cannot merge competing binary edits; coordinate model/corpus replacements.
