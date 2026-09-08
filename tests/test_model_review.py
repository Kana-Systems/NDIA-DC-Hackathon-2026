from types import SimpleNamespace

from app.config import Settings
from app.model_review import ModelReviewService
from app.models import Evidence, GroundingStatus, Severity
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
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
