"""Build and verify an allowlisted teammate bundle; never include local secrets."""

import hashlib
import io
import json
import sqlite3
import subprocess
import tarfile
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/models/legal-bert-cuad"
MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "label_mapping.json",
    "window_config.json",
    "cuad_category_domain_mapping.json",
    "training_provenance.json",
    "evaluation_metrics.json",
    "CUAD_ATTRIBUTION.md",
    "PRETRAINED_ATTRIBUTION.md",
)

INSTRUCTIONS = """# Acquisition Lens teammate handoff

This folder includes the exact pushed `first-test-merge` source snapshot at
COMMIT_PLACEHOLDER plus the completed, selected Legal-BERT model and corrected
FAR/DFARS database. You do NOT need to train a model or ingest the regulations.

## Verify the bundle

Before running anything, compare the ZIP's SHA-256 with the separate SHA256SUMS
file supplied by your teammate. On macOS: `shasum -a 256 ARCHIVE.zip`; on Linux:
`sha256sum ARCHIVE.zip`; on Windows: `Get-FileHash ARCHIVE.zip -Algorithm SHA256`.
HANDOFF_MANIFEST.json contains every included file's checksum and source version.
Checksums detect corruption, not authenticity: obtain them from your teammate.

## Start on macOS or Linux

Install Python (3.12 recommended for broad dependency compatibility) and Node.js
(22.12+ or 24 LTS). Open a terminal INSIDE this extracted folder and run:

```bash
bash scripts/setup-local.sh
bash scripts/run-local.sh
```

Setup also installs the torch and transformers packages needed to USE the
packaged model. These commands install
dependencies and start the app; they do not start training. Installation needs
internet. Use WSL on Windows for these shell commands.

Open http://127.0.0.1:8080/lens/ . Local demo password: `contract-demo`.
These localhost-only demo credentials are not suitable for a public deployment.
Press Ctrl+C in the server terminal to stop the app.

## AWS access for actual LLM reviews

Configure YOUR OWN approved AWS credentials for the project's GovCloud account,
region `us-gov-west-1`, with permission/model access to the existing Bedrock Mantle
GPT-5.6 Terra endpoint. No credentials are supplied. Do not borrow or exchange
personal access keys; ask the account administrator for appropriate access.

The interface can start without AWS access, but real model-assisted review calls
will fail until it is configured. Normal interactive reviews are billable and
are NOT protected by the separate experimental evaluation's $10 cutoff.
Nothing here creates AWS infrastructure. Do not run Terraform, train-sagemaker.sh,
improve-models.py or finish-model-improvements.py merely to start the demo.

## What's included

- Complete committed application/frontend source at the recorded commit.
- artifacts/models/legal-bert-cuad: selected weights, tokenizer, labels,
  configuration, evaluation metrics, provenance and attribution.
- artifacts/models/selected.json: portable relative model selection.
- artifacts/knowledge/federal-v2.sqlite: corrected database used by the launcher.
- artifacts/models/comparison.json: completed baseline/model comparison.
- BUNDLED_CORPUS_MANIFEST.json: source revisions and actual extraction counts.
- SOURCE_ATTRIBUTION.md and the model's attribution/license links.

Not included: .env, AWS keys, browser tokens, uploaded contracts, the raw CUAD
training dataset, Python/Node dependencies, intermediate checkpoints, experimental
models still training, or live evaluation logs. training_args.bin and redundant
model/code scripts are intentionally omitted; inference uses the committed code
and safetensors weights. This archive has no .git directory. To contribute,
clone the repository's first-test-merge branch and copy this bundle's artifacts/
folder into the clone instead of treating this extracted folder as a Git checkout.

## Known limitations

This is the completed baseline, not the unfinished improved candidate. Commercial
CUAD held-out clause-window micro-F1: 0.6933; precision: 0.8727; recall: 0.5750.
These are NOT federal compliance accuracy or legal confidence. Regulatory source
commits are pinned snapshots, not a guarantee that the text is the current law.
The corrected corpus has 5,497 documents, 12,574 passages and 17,920 links.
No organizational approved playbook or independent attorney-reviewed federal
benchmark is included. Human review is required. Model-review inputs currently
have a 30,000-character extracted-text limit. Do not upload sensitive real
contracts without your organization's authorization and appropriate controls.
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    destination = ROOT / "artifacts/handoff"
    destination.mkdir(parents=True, exist_ok=True)
    name = f"acquisition-lens-first-test-merge-{commit[:7]}"
    prefix = name + "/"
    archive_path = destination / f"{name}.zip"
    if archive_path.exists():
        raise FileExistsError(f"Preserve the existing handoff: {archive_path}")
    corpus = ROOT / "artifacts/knowledge/federal-v2.sqlite"
    with sqlite3.connect(f"{corpus.as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("Corpus integrity check failed")
        corpus_manifest = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
        )
    provenance = json.loads((MODEL / "training_provenance.json").read_text())
    selection = json.loads((ROOT / "artifacts/models/selected.json").read_text())
    if (
        selection["model_id"] != provenance["model_id"]
        or selection["model_path"] != "artifacts/models/legal-bert-cuad"
    ):
        raise ValueError("Incumbent selection changed; review the handoff allowlist")
    files = {}
    # Only committed blobs are exported: untracked files and ignored secrets
    # cannot enter through a broad filesystem glob.
    source_tar = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=ROOT)
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as output:

        def add_bytes(relative, data):
            if relative in files:
                raise ValueError(f"Duplicate member {relative}")
            output.writestr(prefix + relative, data)
            files[relative] = {"bytes": len(data), "sha256": sha(data)}

        with tarfile.open(fileobj=io.BytesIO(source_tar)) as source:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or path.name == ".env":
                    raise ValueError(f"Unexpected source member: {member.name}")
                # git archive can export LFS pointers. Add allowlisted, actual
                # artifact bytes below instead, never duplicate pointer entries.
                if path.parts[0] == "artifacts":
                    continue
                add_bytes(member.name, source.extractfile(member).read())
        for filename in MODEL_FILES:
            source = MODEL / filename
            relative = f"artifacts/models/legal-bert-cuad/{filename}"
            output.write(source, prefix + relative)
            files[relative] = {"bytes": source.stat().st_size, "sha256": file_sha(source)}
        relative = "artifacts/knowledge/federal-v2.sqlite"
        output.write(corpus, prefix + relative)
        files[relative] = {"bytes": corpus.stat().st_size, "sha256": file_sha(corpus)}
        for relative in ("artifacts/models/selected.json", "artifacts/models/comparison.json"):
            add_bytes(relative, (ROOT / relative).read_bytes())
        add_bytes("START_HERE.md", INSTRUCTIONS.replace("COMMIT_PLACEHOLDER", commit).encode())
        add_bytes("BUNDLED_CORPUS_MANIFEST.json", json.dumps(corpus_manifest, indent=2).encode())
        add_bytes(
            "SOURCE_ATTRIBUTION.md",
            (
                b"# Bundled regulatory corpus\n\n"
                b"Derived from official GSA FAR and DFARS DITA repositories. "
                b"Source commits and URLs are in BUNDLED_CORPUS_MANIFEST.json "
                b"and each database evidence row. Local changes: text normalization, "
                b"overlapping chunks, SQLite full-text index and reference edges. "
                b"The v2 importer retains main clause text when Alternate sections are present. "
                b"Source retrieval time is not a legal effective date. No Smart Matrix, agency "
                b"supplements, decisions, SAM or USAspending data is bundled.\n\n"
                b"https://github.com/GSA/GSA-Acquisition-FAR\n"
                b"https://github.com/GSA/GSA-Acquisition-DFARS\n\n"
                b"Fine-tuned Legal-BERT weights retain CC BY-SA 4.0; "
                b"CUAD attribution is CC BY 4.0. "
                b"See the model attribution files. Application source retains its LICENSE.\n"
            ),
        )
        manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "source_commit": commit,
            "branch": "first-test-merge",
            "model_id": selection["model_id"],
            "corpus_source_path": "artifacts/knowledge/federal-v2.sqlite",
            "files": files,
        }
        output.writestr(prefix + "HANDOFF_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for relative, expected in files.items():
            with archive.open(prefix + relative) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
                    raise RuntimeError(f"Checksum mismatch: {relative}")
        # Verify inference and database access from the extracted portable bundle,
        # not from the original model directory or developer .env.
        with tempfile.TemporaryDirectory(prefix="lens-handoff-check-") as temporary:
            archive.extractall(temporary)
            check = (
                "import json,sqlite3; from ml.inference import model_fn,predict_fn; "
                "m=model_fn('artifacts/models/legal-bert-cuad'); "
                "r=predict_fn({'texts':['This agreement is governed by the laws of New York.'], "
                "'threshold':0.0},m); "
                "assert len(r['predictions'][0]['labels'])==41; "
                "db=sqlite3.connect('file:artifacts/knowledge/federal-v2.sqlite?mode=ro',"
                "uri=True); "
                "assert db.execute('select count(*) from chunks').fetchone()[0]==12574; "
                "print(json.dumps({'portable_model_loaded':m.model_id,'labels':41,'corpus_passages':12574}))"
            )
            subprocess.run(
                [str(ROOT / ".venv/bin/python"), "-c", check],
                cwd=Path(temporary) / name,
                check=True,
                timeout=180,
            )
    checksum = file_sha(archive_path)
    (destination / f"{name}.SHA256SUMS.txt").write_text(f"{checksum}  {archive_path.name}\n")
    (destination / "START_HERE.md").write_text(INSTRUCTIONS.replace("COMMIT_PLACEHOLDER", commit))
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": checksum,
                "files": len(files) + 1,
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
