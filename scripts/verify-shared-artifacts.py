"""Offline, standard-library checks for the shared inference artifacts."""

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT / "artifacts/models/legal-bert-cuad"
    corpus = ROOT / "artifacts/knowledge/federal-v2.sqlite"
    required = [corpus, model / "model.safetensors"]
    required.extend(
        model / name
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "label_mapping.json",
            "window_config.json",
            "training_provenance.json",
            "evaluation_metrics.json",
            "cuad_category_domain_mapping.json",
            "CUAD_ATTRIBUTION.md",
            "PRETRAINED_ATTRIBUTION.md",
        )
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(
                f"Missing {path.relative_to(ROOT)}. Install Git LFS and run git lfs pull."
            )
        with path.open("rb") as stream:
            if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise SystemExit(
                    f"LFS pointer, not data: {path.relative_to(ROOT)}. Run git lfs pull."
                )
    selection = json.loads((ROOT / "artifacts/models/selected.json").read_text())
    provenance = json.loads((model / "training_provenance.json").read_text())
    if (
        selection["model_path"] != "artifacts/models/legal-bert-cuad"
        or selection["model_id"] != provenance["model_id"]
    ):
        raise SystemExit(
            "Selected model differs from the shared snapshot; review its artifact manifest."
        )
    with sqlite3.connect(f"{corpus.as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SystemExit("Corpus integrity check failed.")
        chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if chunks != 12574:
            raise SystemExit(f"Unexpected corpus snapshot: {chunks} chunks (expected 12574).")
    print(
        f"Shared artifacts ready: {selection['model_id']}; {chunks} regulatory chunks. "
        "No training required."
    )


if __name__ == "__main__":
    main()
