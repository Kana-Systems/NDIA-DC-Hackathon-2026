import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import ml.external_legal_data as external


def _document_for_split(dataset: str, split_name: str) -> str:
    for index in range(10_000):
        document_id = f"document-{index}"
        if external.stable_split(dataset, document_id) == split_name:
            return document_id
    raise AssertionError(f"could not find a document for {split_name}")


def _csv_bytes(document_ids: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "data_type",
            "contract_name",
            "text",
            "answer",
            "label",
            "question",
            "subquestion",
            "text_type",
            "id",
            "category",
        ],
    )
    writer.writeheader()
    for index, document_id in enumerate(document_ids):
        writer.writerow(
            {
                "data_type": "main",
                "contract_name": document_id,
                "text": f"Merger clause unique to {document_id}.",
                "answer": "Yes" if index % 2 else "No",
                "label": str(index % 2),
                "question": "Does the clause permit assignment?",
                "subquestion": "<NONE>",
                "text_type": "Assignment",
                "id": "assignment",
                "category": "Covenants",
            }
        )
    return output.getvalue().encode()


def _write_maud(path: Path, document_ids: list[str]) -> dict[str, str]:
    train = _csv_bytes(document_ids)
    development = _csv_bytes(["publisher-dev"])
    test = _csv_bytes(["publisher-test"])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/MAUD_train.csv", train)
        archive.writestr("data/MAUD_dev.csv", development)
        archive.writestr("data/MAUD_test.csv", test)
        for document_id in document_ids:
            archive.writestr(
                f"data/contracts/{document_id}.txt",
                f"Complete agreement text unique to {document_id}.",
            )
    return {
        "train": hashlib.sha256(train).hexdigest(),
        "development": hashlib.sha256(development).hexdigest(),
        "test": hashlib.sha256(test).hexdigest(),
    }


def _write_contract_nli(path: Path, document_ids: list[str]) -> None:
    dataset = {
        "labels": {
            "nda-1": {
                "short_description": "Survival",
                "hypothesis": "Confidentiality obligations survive termination.",
            }
        },
        "documents": [
            {
                "id": document_id,
                "text": f"Nondisclosure agreement unique to {document_id}.",
                "spans": [[0, 10]],
                "annotation_sets": [
                    {
                        "annotations": {
                            "nda-1": {
                                "choice": "Entailment",
                                "spans": [0],
                            }
                        }
                    }
                ],
            }
            for document_id in document_ids
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("contract-nli/train.json", json.dumps(dataset))
        archive.writestr("contract-nli/dev.json", '{"must_not_be_parsed": true}')
        archive.writestr("contract-nli/test.json", '{"must_not_be_parsed": true}')


def test_near_duplicate_detection_finds_copied_document():
    matches = external.near_duplicate_documents(
        {"cuad-1": "This agreement contains a long and distinctive governing law clause." * 10},
        {"external-1": "This agreement contains a long and distinctive governing law clause." * 10},
    )

    assert matches
    assert matches[0]["candidate_document_id"] == "external-1"
    assert matches[0]["reference_document_id"] == "cuad-1"
    assert matches[0]["shingle_containment"] == 1.0


def test_prepare_uses_only_publisher_train_and_writes_document_disjoint_splits(
    tmp_path, monkeypatch
):
    maud_ids = [
        _document_for_split("MAUD v1", split_name)
        for split_name in ("train", "validation", "selection")
    ]
    contract_ids = [
        _document_for_split("ContractNLI", split_name)
        for split_name in ("train", "validation", "selection")
    ]
    raw = tmp_path / "raw"
    maud_path = raw / "maud" / "maud.zip"
    contract_path = raw / "contractnli" / "contract.zip"
    maud_path.parent.mkdir(parents=True)
    contract_path.parent.mkdir(parents=True)
    inner_hashes = _write_maud(maud_path, maud_ids)
    _write_contract_nli(contract_path, contract_ids)
    sources = (
        external.Source(
            "MAUD v1",
            maud_path.name,
            "https://example.test/maud",
            "test-maud",
            external.sha256_file(maud_path),
            "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        ),
        external.Source(
            "ContractNLI",
            contract_path.name,
            "https://example.test/contractnli",
            "test-contractnli",
            external.sha256_file(contract_path),
            "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/",
        ),
    )
    monkeypatch.setattr(external, "SOURCES", sources)
    monkeypatch.setattr(external, "MAUD_TRAIN_SHA256", inner_hashes["train"])
    monkeypatch.setattr(external, "MAUD_DEV_SHA256", inner_hashes["development"])
    monkeypatch.setattr(external, "MAUD_TEST_SHA256", inner_hashes["test"])
    cuad = tmp_path / "cuad.json"
    cuad.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "cuad-1",
                        "paragraphs": [{"context": "An unrelated distribution agreement."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "processed"
    manifest = external.prepare(raw, output, cuad)

    assert manifest["publisher_development_and_test_labels_opened_by_preparation"] is False
    for dataset in ("maud", "contractnli"):
        split_documents = {}
        for split_name in ("train", "validation", "selection"):
            rows = [
                json.loads(line)
                for line in (output / dataset / f"{split_name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert rows
            split_documents[split_name] = {row["document_id"] for row in rows}
        assert split_documents["train"].isdisjoint(split_documents["validation"])
        assert split_documents["train"].isdisjoint(split_documents["selection"])
        assert split_documents["validation"].isdisjoint(split_documents["selection"])
        assert not (output / dataset / "test.jsonl").exists()
