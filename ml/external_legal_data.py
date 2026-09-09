"""Prepare leak-aware MAUD and ContractNLI transfer-learning datasets.

Only the publishers' training splits are parsed. Official development and test
labels remain unopened so they can be used after model and threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.preprocess_cuad import write_jsonl

SCHEMA_VERSION = "1.0"
MAUD_REVISION = "zenodo:7500064"
MAUD_URL = "https://zenodo.org/records/7500064/files/maud_v1.zip?download=1"
MAUD_SHA256 = "1bbc615913e85c788e8dccce8775079317d0ad011d021526fec0664556a280ee"
MAUD_TRAIN_SHA256 = "bac9f2d034ad487d5398ee2ac1c876679afea509ba8e7f1092112955c6180ff9"
MAUD_DEV_SHA256 = "e977f99abcfd43f785272c23bcf08f8208579a6458d9088d2d5ba64c67e4cc1f"
MAUD_TEST_SHA256 = "afcbc88a4f14e8c14056469446e79f610a6eaf3a1425e61729637bd69251e697"

CONTRACT_NLI_REVISION = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
CONTRACT_NLI_URL = (
    "https://raw.githubusercontent.com/stanfordnlp/contract-nli/"
    f"{CONTRACT_NLI_REVISION}/resources/contract-nli.zip"
)
CONTRACT_NLI_SHA256 = "e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b"

SPLIT_RANGES = {
    "train": range(0, 70),
    "validation": range(70, 85),
    "selection": range(85, 100),
}
NEAR_DUPLICATE_THRESHOLD = 0.80
SHINGLE_WORDS = 12
SHINGLE_STRIDE = 6


@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    url: str
    revision: str
    sha256: str
    license: str
    license_url: str


SOURCES = (
    Source(
        name="MAUD v1",
        filename="maud_v1.zip",
        url=MAUD_URL,
        revision=MAUD_REVISION,
        sha256=MAUD_SHA256,
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
    Source(
        name="ContractNLI",
        filename="contract-nli.zip",
        url=CONTRACT_NLI_URL,
        revision=CONTRACT_NLI_REVISION,
        sha256=CONTRACT_NLI_SHA256,
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def normalized_sha256(value: str) -> str:
    return sha256_bytes(normalized_text(value).encode("utf-8"))


def stable_split(dataset: str, document_id: str) -> str:
    if document_id == "<RARE_ANSWERS>":
        return "train"
    bucket = (
        int(
            hashlib.sha256(f"{dataset}\0{document_id}".encode()).hexdigest()[:8],
            16,
        )
        % 100
    )
    return next(name for name, values in SPLIT_RANGES.items() if bucket in values)


def _download(source: Source, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    request = urllib.request.Request(source.url, headers={"User-Agent": "ndia-legal-data/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    digest = sha256_file(temporary)
    if digest != source.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source.name} SHA-256 mismatch: expected {source.sha256}, received {digest}"
        )
    temporary.replace(path)


def ensure_sources(raw_dir: Path, download: bool, accept_licenses: bool) -> dict[str, Path]:
    if download and not accept_licenses:
        raise ValueError("--download requires --accept-licenses")
    paths: dict[str, Path] = {}
    for source in SOURCES:
        dataset_dir = "maud" if source.name.startswith("MAUD") else "contractnli"
        path = raw_dir / dataset_dir / source.filename
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"missing {source.name} source: {path}")
            _download(source, path)
        digest = sha256_file(path)
        if digest != source.sha256:
            raise RuntimeError(
                f"{source.name} SHA-256 mismatch: expected {source.sha256}, received {digest}"
            )
        paths[source.name] = path
    return paths


def _cuad_documents(path: Path) -> dict[str, str]:
    source = json.loads(path.read_text(encoding="utf-8"))
    documents = {}
    for document in source.get("data", []):
        document_id = str(document.get("title", ""))
        text = "\n".join(
            str(paragraph.get("context", "")) for paragraph in document.get("paragraphs", [])
        )
        if document_id and text.strip():
            documents[document_id] = text
    if not documents:
        raise ValueError("CUAD source contains no documents")
    return documents


def _word_shingles(value: str) -> set[bytes]:
    words = re.findall(r"[a-z]+|\d+", normalized_text(value))
    if len(words) < SHINGLE_WORDS:
        return {hashlib.sha256(" ".join(words).encode("utf-8")).digest()[:8]} if words else set()
    return {
        hashlib.sha256(" ".join(words[index : index + SHINGLE_WORDS]).encode("utf-8")).digest()[:8]
        for index in range(0, len(words) - SHINGLE_WORDS + 1, SHINGLE_STRIDE)
    }


def near_duplicate_documents(
    references: dict[str, str],
    candidates: dict[str, str],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> list[dict[str, Any]]:
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    reference_shingles = {
        document_id: _word_shingles(text) for document_id, text in references.items()
    }
    inverted: dict[bytes, set[str]] = defaultdict(set)
    for document_id, shingles in reference_shingles.items():
        for shingle in shingles:
            inverted[shingle].add(document_id)
    matches = []
    for candidate_id, text in candidates.items():
        candidate_shingles = _word_shingles(text)
        counts: Counter[str] = Counter()
        for shingle in candidate_shingles:
            counts.update(inverted.get(shingle, ()))
        for reference_id, common in counts.items():
            denominator = min(
                len(candidate_shingles),
                len(reference_shingles[reference_id]),
            )
            containment = common / denominator if denominator else 0.0
            if containment >= threshold:
                matches.append(
                    {
                        "candidate_document_id": candidate_id,
                        "reference_document_id": reference_id,
                        "shingle_containment": containment,
                        "shared_shingles": common,
                    }
                )
    return sorted(
        matches,
        key=lambda item: (
            -item["shingle_containment"],
            item["candidate_document_id"],
            item["reference_document_id"],
        ),
    )


def _maud_contracts(archive: zipfile.ZipFile) -> dict[str, str]:
    contracts = {}
    for name in archive.namelist():
        if name.startswith("data/contracts/") and name.endswith(".txt"):
            document_id = Path(name).stem
            contracts[document_id] = archive.read(name).decode("utf-8", errors="replace")
    if not contracts:
        raise ValueError("MAUD archive contains no contract texts")
    return contracts


def _maud_records(archive: zipfile.ZipFile, excluded_documents: set[str]) -> list[dict[str, Any]]:
    train_bytes = archive.read("data/MAUD_train.csv")
    if sha256_bytes(train_bytes) != MAUD_TRAIN_SHA256:
        raise RuntimeError("MAUD training split SHA-256 does not match the pinned release")
    # Hash but intentionally do not parse the publisher's dev and test splits.
    if sha256_bytes(archive.read("data/MAUD_dev.csv")) != MAUD_DEV_SHA256:
        raise RuntimeError("MAUD development split SHA-256 does not match the pinned release")
    if sha256_bytes(archive.read("data/MAUD_test.csv")) != MAUD_TEST_SHA256:
        raise RuntimeError("MAUD test split SHA-256 does not match the pinned release")
    rows = csv.DictReader(io.StringIO(train_bytes.decode("utf-8-sig")))
    records = []
    seen = set()
    for row_number, row in enumerate(rows, start=2):
        document_id = row["contract_name"].strip()
        if document_id in excluded_documents:
            continue
        task_id = row["id"].strip()
        label = row["label"].strip()
        subquestion = row["subquestion"].strip()
        prompt_parts = [
            "Task: MAUD merger-agreement review",
            f"Question: {row['question'].strip()}",
        ]
        if subquestion and subquestion != "<NONE>":
            prompt_parts.append(f"Answer option under review: {subquestion}")
        prompt_parts.append(f"Clause:\n{row['text'].strip()}")
        text = "\n".join(prompt_parts)
        dedupe_key = (document_id, task_id, label, normalized_sha256(text))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(
            {
                "id": f"maud:train:{row_number}",
                "document_id": document_id,
                "task_id": task_id,
                "text": text,
                "labels": [label],
                "answer": row["answer"].strip(),
                "source_data_type": row["data_type"].strip(),
            }
        )
    if not records:
        raise ValueError("MAUD training split produced no records")
    return records


def _contract_nli_training(archive: zipfile.ZipFile) -> tuple[list[dict[str, Any]], dict[str, str]]:
    dataset = json.loads(archive.read("contract-nli/train.json"))
    labels = dataset.get("labels", {})
    records = []
    documents = {}
    for document in dataset.get("documents", []):
        document_id = str(document["id"])
        document_text = str(document["text"])
        documents[document_id] = document_text
        annotations = document["annotation_sets"][0]["annotations"]
        for hypothesis_id, annotation in sorted(annotations.items()):
            hypothesis = labels[hypothesis_id]["hypothesis"]
            records.append(
                {
                    "id": f"contractnli:train:{document_id}:{hypothesis_id}",
                    "document_id": document_id,
                    "task_id": hypothesis_id,
                    "text": (
                        "Task: ContractNLI document-level inference\n"
                        f"Hypothesis: {hypothesis}\n"
                        f"Contract:\n{document_text}"
                    ),
                    "labels": [annotation["choice"]],
                    "evidence_span_count": len(annotation.get("spans", [])),
                }
            )
    if len(records) != len(documents) * len(labels):
        raise ValueError("ContractNLI training annotations are incomplete")
    return records, documents


def _write_task(
    output_dir: Path,
    records: list[dict[str, Any]],
    labels: list[str],
    source: Source,
    excluded_documents: set[str],
    overlap_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    splits = {name: [] for name in SPLIT_RANGES}
    document_splits: dict[str, str] = {}
    for record in records:
        document_id = record["document_id"]
        split_name = stable_split(source.name, document_id)
        previous = document_splits.setdefault(document_id, split_name)
        if previous != split_name:
            raise RuntimeError(f"document leaked across splits: {document_id}")
        splits[split_name].append(record)
    if any(not rows for rows in splits.values()):
        raise ValueError(f"{source.name} generated an empty split")
    split_document_ids = {
        name: {record["document_id"] for record in rows} for name, rows in splits.items()
    }
    if any(
        split_document_ids[left] & split_document_ids[right]
        for left in split_document_ids
        for right in split_document_ids
        if left < right
    ):
        raise RuntimeError(f"{source.name} document splits overlap")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )
    maximum_chars = max(len(record["text"]) for record in records)
    window_config = {
        "schema_version": "1.0",
        "strategy": "task_prompt_with_full_available_context",
        "window_chars": maximum_chars,
        "stride_chars": maximum_chars,
    }
    atomic_json(output_dir / "window_config.json", window_config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "name": source.name,
            "url": source.url,
            "revision": source.revision,
            "sha256": source.sha256,
            "license": source.license,
            "license_url": source.license_url,
        },
        "policy": {
            "parsed_publisher_split": "train",
            "publisher_dev_labels_opened_by_preparation": False,
            "publisher_test_labels_opened_by_preparation": False,
            "partition": (
                "stable document hash: 70% train, 15% checkpoint validation, 15% selection"
            ),
            "selection_is_not_trainer_validation": True,
            "publisher_dev_and_test_not_emitted_or_used": True,
        },
        "record_counts": {name: len(rows) for name, rows in splits.items()},
        "document_counts": {
            name: len(document_ids) for name, document_ids in split_document_ids.items()
        },
        "excluded_cuad_overlap_documents": sorted(excluded_documents),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "cuad_overlap_matches": overlap_matches,
        "output_sha256": {
            f"{name}.jsonl": sha256_file(output_dir / f"{name}.jsonl") for name in splits
        },
        "labels_sha256": sha256_file(output_dir / "labels.json"),
        "window_config_sha256": sha256_file(output_dir / "window_config.json"),
        "preparation_code_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(output_dir / "preparation.json", manifest)
    return manifest


def prepare(
    raw_dir: Path = Path("ml/data/external/raw"),
    output_dir: Path = Path("ml/data/external/processed"),
    cuad_path: Path = Path("ml/data/CUAD_v1.json"),
    *,
    download: bool = False,
    accept_licenses: bool = False,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    cuad_path = Path(cuad_path)
    if not cuad_path.is_file():
        raise FileNotFoundError(f"CUAD source not found: {cuad_path}")
    source_paths = ensure_sources(raw_dir, download, accept_licenses)
    cuad_documents = _cuad_documents(cuad_path)

    with zipfile.ZipFile(source_paths["MAUD v1"]) as maud_archive:
        maud_documents = _maud_contracts(maud_archive)
        maud_overlap = near_duplicate_documents(cuad_documents, maud_documents)
        maud_excluded = {match["candidate_document_id"] for match in maud_overlap}
        maud_records = _maud_records(maud_archive, maud_excluded)

    with zipfile.ZipFile(source_paths["ContractNLI"]) as contract_archive:
        contract_records, contract_documents = _contract_nli_training(contract_archive)
        contract_overlap = near_duplicate_documents(cuad_documents, contract_documents)
        contract_excluded = {match["candidate_document_id"] for match in contract_overlap}
        contract_records = [
            record for record in contract_records if record["document_id"] not in contract_excluded
        ]

    maud_manifest = _write_task(
        output_dir / "maud",
        maud_records,
        [str(index) for index in range(10)],
        SOURCES[0],
        maud_excluded,
        maud_overlap,
    )
    contract_manifest = _write_task(
        output_dir / "contractnli",
        contract_records,
        ["Contradiction", "Entailment", "NotMentioned"],
        SOURCES[1],
        contract_excluded,
        contract_overlap,
    )
    combined = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "publisher_development_and_test_labels_opened_by_preparation": False,
        "maud_preparation_sha256": sha256_file(output_dir / "maud" / "preparation.json"),
        "contractnli_preparation_sha256": sha256_file(
            output_dir / "contractnli" / "preparation.json"
        ),
        "maud": maud_manifest,
        "contractnli": contract_manifest,
    }
    atomic_json(output_dir / "manifest.json", combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("ml/data/external/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data/external/processed"),
    )
    parser.add_argument("--cuad-path", type=Path, default=Path("ml/data/CUAD_v1.json"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--accept-licenses", action="store_true")
    args = parser.parse_args()
    manifest = prepare(
        args.raw_dir,
        args.output_dir,
        args.cuad_path,
        download=args.download,
        accept_licenses=args.accept_licenses,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
