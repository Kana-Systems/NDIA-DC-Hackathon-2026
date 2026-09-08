import inspect

from app.config import Settings
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
from app.service import ReviewService
from app.ui import (
    _bind_run_review,
    _finding_detail,
    _finding_selector_data,
    _safe_citation_link,
)


def _report_data():
    settings = Settings()
    document = DocumentParser(settings).parse("sample.docx", sample_contract_bytes())
    return ReviewService(settings).review(document, sample_metadata()).model_dump(mode="json")


def test_finding_selector_details_include_grounded_judge_context() -> None:
    report_data = _report_data()
    finding = next(item for item in report_data["findings"] if item["rule_id"] == "GC-006")

    detail = _finding_detail(report_data, finding["finding_id"])

    assert "Document text and location" in detail
    assert "Page 1, paragraph 4" in detail
    assert "Payment will be made 90 days" in detail
    assert "Explanation" in detail
    assert "Proposed replacement / guidance" in detail
    assert "Classifier confidence:" in detail
    assert "acquisition.gov/far/part-32" in detail
    assert "EV-FAR-32" in detail


def test_dfars_finding_uses_canonical_clause_and_prescription_links() -> None:
    report_data = _report_data()
    finding = next(item for item in report_data["findings"] if item["rule_id"] == "GC-004")

    detail = _finding_detail(report_data, finding["finding_id"])

    assert "204.7304-solicitation-provision-and-contract-clauses." in detail
    assert (
        "252.204-7012-safeguarding-covered-defense-information-and-cyber-incident-reporting."
    ) in detail
    assert "EV-DFARS-204.7304-C" in detail
    assert "EV-DFARS-7012" in detail


def test_selector_respects_severity_filter() -> None:
    report_data = _report_data()

    choices, selected = _finding_selector_data(report_data, "critical")

    assert choices
    assert all("CRITICAL" in label for label, _ in choices)
    assert selected == choices[0][1]


def test_citation_link_rejects_unsafe_url_scheme() -> None:
    rendered = _safe_citation_link("javascript:alert(1)", "Official [source]")

    assert "javascript:" not in rendered
    assert rendered == r"Official \[source\]"


def test_bound_review_remains_a_generator_function_for_gradio() -> None:
    callback = _bind_run_review(Settings())

    assert inspect.isgeneratorfunction(callback)
