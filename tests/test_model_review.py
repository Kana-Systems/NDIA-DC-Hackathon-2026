import io
import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.model_review import ModelReviewService, SageMakerClassifier
from app.models import Evidence, GroundingStatus, Severity
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
from app.workspace import parsed_text
from ml.inference import input_fn, output_fn
from ml.preprocess_cuad import examples


def test_long_cuad_answer_is_windowed_without_losing_labels():
    text = "legal obligation " * 250
    data = {
        "data": [
            {
                "title": "contract-1",
                "paragraphs": [
                    {
                        "context": text,
                        "qas": [
                            {
                                "id": "contract-1__Liability",
                                "answers": [
                                    {"text": text, "answer_start": 0},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    windows = examples(data, window_chars=240, stride_chars=120)
    assert len(windows) > 1
    assert all(len(window["text"]) <= 240 for window in windows)
    assert all(window["labels"] == ["liability"] for window in windows)
    assert windows[-1]["end_char"] == len(text)


def test_model_generated_citations_and_locations_are_validated(monkeypatch):
    settings = Settings()
    document = DocumentParser(settings).parse("sample.docx", sample_contract_bytes())
    service = ModelReviewService.__new__(ModelReviewService)
    service.settings = settings
    service.model = SimpleNamespace(model_id="trained-test-model")
    evidence = Evidence(evidence_id="known", source="FAR", title="Source", excerpt="Source text")
    service.retrieval = SimpleNamespace(
        retrieve=lambda queries: [evidence],
        manifest=lambda: {},
        relations=lambda ids: [],
    )
    service.llm = SimpleNamespace(
        request_json=lambda *_args, **_kwargs: {
            "queries": ["termination"],
            "executive_summary": "Potential issues",
            "findings": [
                {
                    "title": "Invented citation",
                    "severity": "critical",
                    "description": "Observation",
                    "recommendation": "Review",
                    "citation_ids": ["known", "fabricated"],
                    "segment_ids": ["nonexistent"],
                }
            ],
        }
    )
    monkeypatch.setattr(
        "app.model_review.predict_fn",
        lambda payload, model: {
            "predictions": [
                {"model_id": "trained-test-model", "labels": []} for _ in payload["texts"]
            ],
        },
    )
    report = service.review(document, sample_metadata())
    assert report.findings[0].severity == Severity.INFO
    assert report.findings[0].grounding_status == GroundingStatus.UNVERIFIED
    assert report.findings[0].citation_ids == ["known"]
    assert report.findings[0].locations == []
    assert "1 are unverified" in report.executive_summary


def test_txt_review_skips_blank_blocks_and_preserves_evidence_locations():
    text = (
        "\n\nTermination for convenience applies.\n\n\n\n \t \n\nPayment is due in thirty days.\n\n"
    )
    document = parsed_text("sponsorship-agreement.txt", text)
    original = document.model_dump()
    model_id = "Llama-3.1-CUAD-r128-ensemble-3seed"

    class Client:
        def invoke_endpoint(self, **kwargs):
            # Use the deployed artifact's input validator, which rejects blank blocks.
            payload = input_fn(kwargs["Body"], kwargs["ContentType"])
            assert payload["texts"] == [
                "Termination for convenience applies.",
                "Payment is due in thirty days.",
            ]
            result = {
                "predictions": [
                    {"model_id": model_id, "labels": [{"label": label, "score": 0.9}]}
                    for label in ("termination_for_convenience", "payment_terms")
                ]
            }
            return {"Body": io.BytesIO(json.dumps(output_fn(result, "application/json")).encode())}

    settings = Settings(
        classifier_endpoint_name="test-endpoint", classifier_endpoint_model_id=model_id
    )
    service = ModelReviewService.__new__(ModelReviewService)
    service.settings = settings
    service.endpoint = SageMakerClassifier(settings, client=Client())
    service.classifier_model_id = model_id
    evidence = Evidence(evidence_id="known", source="FAR", title="Source", excerpt="Source text")
    service.retrieval = SimpleNamespace(
        retrieve=lambda queries: [evidence], manifest=lambda: {}, relations=lambda ids: []
    )
    service.llm = SimpleNamespace(
        request_json=lambda *_args, **_kwargs: {
            "queries": ["termination"],
            "executive_summary": "Review the termination terms.",
            "findings": [
                {
                    "title": "Check termination terms",
                    "severity": "info",
                    "description": "Confirm the termination obligation.",
                    "recommendation": "Inspect the cited passage.",
                    "citation_ids": ["known"],
                    "segment_ids": ["text-2"],
                }
            ],
        }
    )
    report = service.review(document, sample_metadata())
    assert document.model_dump() == original
    assert document.text == text
    assert report.findings[0].locations == [document.segments[1].location]
    learned = [
        clause for clause in report.clause_inventory if clause.classifier_model_id == model_id
    ]
    assert [(clause.segment_id, clause.location.paragraph) for clause in learned] == [
        ("text-2", 2),
        ("text-5", 5),
    ]


def test_blank_contract_is_rejected_before_model_calls():
    service = ModelReviewService.__new__(ModelReviewService)
    with pytest.raises(ValueError, match="no reviewable text"):
        service.review(parsed_text("empty.txt", "\n\n \t \n\n"), sample_metadata())
