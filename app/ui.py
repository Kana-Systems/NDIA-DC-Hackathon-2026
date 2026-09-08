"""Judge-facing Gradio workflow."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import gradio as gr

from app.config import Settings
from app.intelligence import (
    CitedGenerationService,
    IntelligenceRepository,
    ingestion_operational_status,
)
from app.models import (
    AcquisitionMetadata,
    AcquisitionStage,
    AnalystDecisionRequest,
    ContractType,
    EntityCandidate,
    EntityResolutionRequest,
    GenerationMode,
    IntelligenceQuery,
    PrincipalContext,
    ReviewDecision,
    ReviewReport,
    Severity,
    TargetObjectDraftRequest,
)
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
from app.service import ReviewService

ui_intelligence_repository = IntelligenceRepository()


def _ui_principal(settings: Settings) -> PrincipalContext:
    return PrincipalContext(
        subject=settings.gradio_username,
        groups=["contract-reviewers", "mission-analysts"],
        security_domain=settings.security_domain,
        scopes=[
            "rag:query",
            "entities:read",
            "objects:draft",
            "objects:review",
        ],
    )


def _run_intelligence_query(
    settings: Settings,
    query: str,
    mode: str,
) -> tuple[str, dict[str, Any]]:
    try:
        response = CitedGenerationService(settings).generate(
            IntelligenceQuery(query=query, mode=GenerationMode(mode)),
            _ui_principal(settings),
        )
    except Exception:
        return "The grounded intelligence request could not be completed.", {}
    citations = (
        "\n".join(
            f"- `{item.evidence_id}` — {_markdown_text(item.title)}" for item in response.evidence
        )
        or "- No authorized evidence"
    )
    return (
        f"## {response.mode.value.title()}\n{_markdown_text(response.answer)}\n\n"
        f"### Authorized citations\n{citations}\n\n"
        f"**Synthesis:** `{response.synthesis_mode}`",
        response.model_dump(mode="json"),
    )


def _resolve_entity_from_ui(
    settings: Settings,
    name: str,
    entity_type: str,
    attributes_json: str,
    citation_id: str,
) -> tuple[str, dict[str, Any]]:
    try:
        attributes = json.loads(attributes_json or "{}")
        if not isinstance(attributes, dict):
            raise ValueError
        provenance = []
        if citation_id.strip():
            from app.models import ProvenanceLink

            provenance = [
                ProvenanceLink(
                    citation_id=citation_id.strip(),
                    document_id="analyst-selected-source",
                )
            ]
        entities = ui_intelligence_repository.resolve(
            EntityResolutionRequest(
                candidates=[
                    EntityCandidate(
                        name=name,
                        entity_type=entity_type,
                        attributes=attributes,
                        provenance=provenance,
                    )
                ]
            ).candidates
        )
        entity = entities[0]
        return (
            f"Resolved `{_markdown_text(entity.canonical_name)}` as "
            f"`{entity.entity_id}`. Status: **{entity.review_status.value}**.",
            entity.model_dump(mode="json"),
        )
    except (ValueError, TypeError):
        return "Enter a name, type, and a JSON object of attributes.", {}


def _draft_target_from_ui(
    settings: Settings,
    entity_id: str,
    object_type: str,
    requested_fields: str,
) -> tuple[str, dict[str, Any]]:
    try:
        target = ui_intelligence_repository.create_target_object(
            TargetObjectDraftRequest(
                object_type=object_type,
                entity_id=entity_id.strip(),
                requested_fields=[
                    value.strip() for value in requested_fields.split(",") if value.strip()
                ],
            ),
            _ui_principal(settings),
        )
        return (
            f"Draft `{target.object_id}` created. An analyst must approve it before JSON export.",
            target.model_dump(mode="json"),
        )
    except (KeyError, PermissionError, ValueError):
        return "Create an entity first and provide its entity ID.", {}


def _decide_entity_from_ui(
    settings: Settings,
    entity_id: str,
    decision: str,
    note: str,
) -> tuple[str, dict[str, Any]]:
    try:
        entity = ui_intelligence_repository.decide_entity(
            entity_id.strip(),
            AnalystDecisionRequest(
                decision=ReviewDecision(decision),
                note=note,
            ),
            _ui_principal(settings),
        )
        return (
            f"Entity `{entity.entity_id}` is now **{entity.review_status.value}**.",
            entity.model_dump(mode="json"),
        )
    except (KeyError, PermissionError, ValueError):
        return "The entity decision could not be recorded.", {}


def _decide_target_from_ui(
    settings: Settings,
    object_id: str,
    decision: str,
    note: str,
) -> tuple[str, dict[str, Any]]:
    try:
        target = ui_intelligence_repository.decide(
            object_id.strip(),
            AnalystDecisionRequest(
                decision=ReviewDecision(decision),
                note=note,
            ),
            _ui_principal(settings),
        )
        return (
            f"Object `{target.object_id}` is now **{target.status.value}**. "
            "Approved objects are available through the JSON export API.",
            target.model_dump(mode="json"),
        )
    except (KeyError, PermissionError, ValueError):
        return (
            "The decision was rejected. Verify the object ID and ensure every "
            "populated field has source evidence.",
            {},
        )


def _severity_markdown(report: ReviewReport) -> str:
    counts = report.severity_counts()
    labels = {
        Severity.CRITICAL: "🔴 Critical",
        Severity.HIGH: "🟠 High",
        Severity.MEDIUM: "🟡 Medium",
        Severity.LOW: "🔵 Low",
        Severity.INFO: "⚪ Info",
    }
    severity_lines = "  \n".join(
        f"### {labels[severity]}: {counts[severity.value]}" for severity in Severity
    )
    ablation = report.ablation_summary
    return (
        f"{severity_lines}\n\n"
        "### Ablation summary\n"
        f"- Deterministic keyword clauses: **{ablation.deterministic_clause_count}**\n"
        f"- Classifier-assisted clauses: **{ablation.classifier_assisted_clause_count}** "
        f"(+{ablation.classifier_added_clause_count})\n"
        f"- Findings without/with classifier assistance: "
        f"**{ablation.deterministic_finding_count} / "
        f"{ablation.classifier_assisted_finding_count}**\n"
        f"- Retrieved evidence records: **{ablation.retrieved_evidence_count}**\n"
        f"- Classifier models: **{', '.join(report.classifier_model_ids) or 'none'}**"
    )


def _findings_markdown(report: ReviewReport, severity: str = "all") -> str:
    filtered = [
        finding
        for finding in report.findings
        if severity == "all" or finding.severity.value == severity
    ]
    if not filtered:
        return "No rule findings were identified. Human review is still required."
    blocks = []
    for finding in filtered:
        locations = ", ".join(location.label for location in finding.locations) or "Document-wide"
        citations = ", ".join(f"`{item}`" for item in finding.citation_ids) or "None"
        blocks.append(
            f"### {finding.finding_id} · {finding.severity.value.upper()} · {finding.title}\n"
            f"{finding.description}\n\n"
            f"**Location:** {locations}  \n"
            f"**Evidence:** {citations}  \n"
            f"**Grounding:** {finding.grounding_status.value.upper()}  \n"
            f"**Recommended action:** {finding.recommendation}"
        )
    return "\n\n---\n\n".join(blocks)


def _filter_findings(report_data: dict[str, Any] | None, severity: str) -> str:
    if not report_data:
        return "Run a review to view findings."
    try:
        return _findings_markdown(ReviewReport.model_validate(report_data), severity)
    except ValueError:
        return "Finding details are unavailable."


def _markdown_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in ("`", "*", "_", "[", "]", "(", ")", "<", ">", "#"):
        escaped = escaped.replace(character, f"\\{character}")
    return escaped


def _safe_citation_link(url: str | None, label: str) -> str:
    if not url or any(character in url for character in ("\r", "\n", "\x00")):
        return _markdown_text(label)
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _markdown_text(label)
    safe_url = quote(url, safe=":/?#[]@!$&'*+,;=%-._~")
    return f"[{_markdown_text(label)}]({safe_url})"


def _finding_selector_data(
    report_data: dict[str, Any] | None,
    severity: str = "all",
) -> tuple[list[tuple[str, str]], str | None]:
    if not report_data:
        return [], None
    try:
        report = ReviewReport.model_validate(report_data)
    except ValueError:
        return [], None
    choices = [
        (
            f"{finding.finding_id} · {finding.severity.value.upper()} · {finding.title}",
            finding.finding_id,
        )
        for finding in report.findings
        if severity == "all" or finding.severity.value == severity
    ]
    return choices, choices[0][1] if choices else None


def _finding_detail(
    report_data: dict[str, Any] | None,
    finding_id: str | None,
) -> str:
    if not report_data or not finding_id:
        return "Select a finding to view grounded details."
    try:
        report = ReviewReport.model_validate(report_data)
    except ValueError:
        return "Finding details are unavailable."
    finding = next(
        (item for item in report.findings if item.finding_id == finding_id),
        None,
    )
    if finding is None:
        return "Finding details are unavailable."
    evidence_by_id = {item.evidence_id: item for item in report.evidence}
    contexts = (
        "\n\n".join(
            f"**{_markdown_text(context.location.label)}**\n\n> {_markdown_text(context.text)}"
            for context in finding.document_context
        )
        or "No specific document passage was associated with this finding."
    )
    confidence = (
        f"{finding.classifier_confidence:.1%}"
        if finding.classifier_confidence is not None
        else "Not available for this rule finding"
    )
    citations = []
    for citation_id in finding.citation_ids:
        evidence = evidence_by_id.get(citation_id)
        if evidence is None:
            continue
        label = f"{evidence.title} — {evidence.source} ({evidence.evidence_id})"
        citations.append(f"- {_safe_citation_link(evidence.url, label)}")
    citation_text = "\n".join(citations) or "No resolved authoritative citation."
    return (
        f"## {_markdown_text(finding.title)}\n"
        f"**Severity:** {finding.severity.value.upper()}  \n"
        f"**Grounding:** {finding.grounding_status.value.upper()}  \n"
        f"**Classifier confidence:** {confidence}  \n"
        f"**Classifier model(s):** "
        f"{_markdown_text(', '.join(finding.classifier_model_ids) or 'Not applicable')}\n\n"
        f"### Document text and location\n{contexts}\n\n"
        f"### Explanation\n{_markdown_text(finding.description)}\n\n"
        f"### Proposed replacement / guidance\n"
        f"{_markdown_text(finding.recommendation)}\n\n"
        f"### Authoritative citations\n{citation_text}"
    )


def _initialize_finding_panel(
    report_data: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    choices, selected = _finding_selector_data(report_data)
    return (
        gr.update(choices=choices, value=selected),
        _finding_detail(report_data, selected),
    )


def _filter_and_select(
    report_data: dict[str, Any] | None,
    severity: str,
) -> tuple[str, dict[str, Any], str]:
    choices, selected = _finding_selector_data(report_data, severity)
    return (
        _filter_findings(report_data, severity),
        gr.update(choices=choices, value=selected),
        _finding_detail(report_data, selected),
    )


def _run_review(
    settings: Settings,
    file_value: Any,
    use_sample: bool,
    agency: str,
    solicitation_number: str,
    contract_type: str,
    estimated_value: float,
    set_aside: str,
    commercial_product: bool,
    cots_only: bool,
    performance_months: int,
    place_of_performance: str,
    acquisition_stage: str,
) -> Iterator[tuple[str, str, str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    empty: tuple[str, str, str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
    empty = ("", "", "", [], [], {})
    yield ("1/4 Validating document and metadata…", *empty[1:])
    parser = DocumentParser(settings)
    path: Path | None = None
    try:
        if use_sample:
            filename = "sample-contract.docx"
            data = sample_contract_bytes()
            metadata = sample_metadata()
        else:
            if not file_value:
                raise ValueError("Upload a PDF or DOCX, or select the sample contract.")
            path = Path(str(file_value))
            if path.suffix.lower() not in settings.allowed_extensions:
                raise ValueError("Only PDF and DOCX uploads are accepted.")
            if path.stat().st_size > settings.max_upload_bytes:
                raise ValueError(f"Upload exceeds the {settings.max_upload_mb} MB limit.")
            data = path.read_bytes()
            filename = path.name
            metadata = AcquisitionMetadata(
                agency=agency,
                solicitation_number=solicitation_number,
                contract_type=ContractType(contract_type),
                estimated_value=estimated_value,
                set_aside=set_aside,
                commercial_product=commercial_product,
                cots_only=cots_only,
                performance_months=performance_months,
                place_of_performance=place_of_performance,
                acquisition_stage=AcquisitionStage(acquisition_stage),
            )
        yield ("2/4 Extracting located text…", *empty[1:])
        document = parser.parse_isolated(filename, data)
        yield ("3/4 Classifying clauses and applying deterministic rules…", *empty[1:])
        report = ReviewService(settings).review(document, metadata)
        yield (
            f"4/4 Review complete · synthesis: {report.synthesis_mode}",
            _severity_markdown(report),
            f"## Executive summary\n{report.executive_summary}\n\n" + _findings_markdown(report),
            [item.model_dump(mode="json") for item in report.clause_status_inventory],
            [item.model_dump(mode="json") for item in report.evidence],
            report.model_dump(mode="json"),
        )
    except Exception:
        # Do not expose parser, filesystem, AWS, or source-document details in the UI.
        yield (
            "Review could not be completed. Verify the file and metadata, then retry.",
            "",
            "",
            [],
            [],
            {},
        )
    finally:
        # Gradio copies uploads into its cache; remove only that temporary copy.
        if path and "gradio" in str(path.parent).lower():
            path.unlink(missing_ok=True)


def _bind_run_review(settings: Settings) -> Any:
    """Bind settings while preserving generator-function semantics for Gradio."""

    def run_review(
        *args: Any,
    ) -> Iterator[
        tuple[
            str,
            str,
            str,
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, Any],
        ]
    ]:
        yield from _run_review(settings, *args)

    return run_review


def build_ui(settings: Settings) -> gr.Blocks:
    with gr.Blocks(title=settings.app_name, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Government Contract Review\n"
            "Traceable pre-review for acquisition teams. Results support—not replace—"
            "contracting officer and counsel judgment."
        )
        use_sample = gr.Checkbox(label="Use built-in sample contract", value=False)
        upload = gr.File(
            label=f"Contract (.pdf or .docx, max {settings.max_upload_mb} MB)",
            file_types=[".pdf", ".docx"],
            type="filepath",
        )
        with gr.Row():
            agency = gr.Textbox(label="Agency", value="Department of Defense")
            solicitation = gr.Textbox(label="Solicitation number", value="DEMO-2026-001")
            contract_type = gr.Dropdown(
                choices=[item.value for item in ContractType],
                value=ContractType.FIRM_FIXED_PRICE.value,
                label="Contract type",
            )
        with gr.Row():
            value = gr.Number(label="Estimated value ($)", value=1_500_000, minimum=0)
            set_aside = gr.Textbox(label="Set-aside", value="Small Business")
            duration = gr.Slider(1, 240, value=12, step=1, label="Performance period (months)")
            commercial = gr.Checkbox(label="Commercial product/service")
            cots_only = gr.Checkbox(label="Solely commercial off-the-shelf (COTS)")
        with gr.Row():
            place_of_performance = gr.Textbox(
                label="Place of performance",
                value="Arlington, Virginia",
            )
            acquisition_stage = gr.Dropdown(
                choices=[item.value for item in AcquisitionStage],
                value=AcquisitionStage.SOLICITATION.value,
                label="Acquisition stage",
            )
        review_button = gr.Button("Run contract review", variant="primary")
        progress = gr.Markdown("Ready.")
        severity = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("Findings"):
                severity_filter = gr.Dropdown(
                    choices=["all", *[item.value for item in Severity]],
                    value="all",
                    label="Filter by severity",
                )
                findings = gr.Markdown()
                finding_selector = gr.Dropdown(
                    choices=[],
                    label="Select finding details",
                )
                finding_detail = gr.Markdown("Select a finding to view grounded details.")
            with gr.Tab("Clause inventory"):
                inventory = gr.JSON()
            with gr.Tab("Retrieved evidence"):
                evidence = gr.JSON()
            with gr.Tab("JSON report"):
                report = gr.JSON()
        review_event = review_button.click(
            fn=_bind_run_review(settings),
            inputs=[
                upload,
                use_sample,
                agency,
                solicitation,
                contract_type,
                value,
                set_aside,
                commercial,
                cots_only,
                duration,
                place_of_performance,
                acquisition_stage,
            ],
            outputs=[progress, severity, findings, inventory, evidence, report],
        )
        review_event.then(
            fn=_initialize_finding_panel,
            inputs=report,
            outputs=[finding_selector, finding_detail],
        )
        severity_filter.change(
            fn=_filter_and_select,
            inputs=[report, severity_filter],
            outputs=[findings, finding_selector, finding_detail],
        )
        finding_selector.change(
            fn=_finding_detail,
            inputs=[report, finding_selector],
            outputs=finding_detail,
        )
        gr.Markdown(
            "## Joint Staff J2 intelligence workflows\n"
            "These modules reuse the same authorization and citation trust boundary. "
            "Generated content and structured objects remain analyst-review drafts."
        )
        with gr.Tabs():
            with gr.Tab("Ask / Draft"):
                rag_query = gr.Textbox(
                    label="Natural-language request",
                    value="Summarize the provenance requirements for foundational data.",
                    lines=3,
                )
                rag_mode = gr.Dropdown(
                    choices=[item.value for item in GenerationMode],
                    value=GenerationMode.ANSWER.value,
                    label="Output mode",
                )
                rag_button = gr.Button("Generate grounded response", variant="primary")
                rag_answer = gr.Markdown()
                rag_report = gr.JSON(label="Grounded response")
                rag_button.click(
                    fn=lambda query, mode: _run_intelligence_query(settings, query, mode),
                    inputs=[rag_query, rag_mode],
                    outputs=[rag_answer, rag_report],
                )
            with gr.Tab("Sources / Ingestion"):
                gr.Markdown(
                    "The deployment worker performs versioned, idempotent ingestion "
                    "from Microsoft Graph/shared drives and automatically loads a "
                    "120-document public/synthetic fixture corpus. Connector permissions "
                    "are preserved as retrieval ACLs; source text is not written to logs."
                )
                source_status_button = gr.Button("Refresh ingestion status")
                source_status = gr.JSON(label="Operational ingestion status")
                source_status_button.click(
                    fn=lambda: ingestion_operational_status(settings),
                    inputs=[],
                    outputs=source_status,
                )
            with gr.Tab("Foundational Intelligence"):
                entity_name = gr.Textbox(label="Entity name")
                entity_type = gr.Textbox(label="Entity type", value="organization")
                entity_attributes = gr.Code(
                    label="Structured attributes",
                    language="json",
                    value='{"country": "US", "status": "active"}',
                )
                entity_citation = gr.Textbox(
                    label="Supporting citation ID",
                    value="demo:foundational-data:1",
                )
                entity_button = gr.Button("Resolve and stage entity")
                entity_status = gr.Markdown()
                entity_result = gr.JSON(label="Entity and provenance")
                entity_button.click(
                    fn=lambda name, kind, attributes, citation: _resolve_entity_from_ui(
                        settings,
                        name,
                        kind,
                        attributes,
                        citation,
                    ),
                    inputs=[
                        entity_name,
                        entity_type,
                        entity_attributes,
                        entity_citation,
                    ],
                    outputs=[entity_status, entity_result],
                )
                entity_decision_id = gr.Textbox(label="Resolved entity ID")
                entity_decision = gr.Dropdown(
                    choices=[
                        ReviewDecision.APPROVED.value,
                        ReviewDecision.REJECTED.value,
                    ],
                    value=ReviewDecision.APPROVED.value,
                    label="Quality-control decision",
                )
                entity_note = gr.Textbox(label="Analyst note", lines=2)
                entity_decision_button = gr.Button("Record entity decision")
                entity_decision_button.click(
                    fn=lambda entity_id, decision, note: _decide_entity_from_ui(
                        settings,
                        entity_id,
                        decision,
                        note,
                    ),
                    inputs=[entity_decision_id, entity_decision, entity_note],
                    outputs=[entity_status, entity_result],
                )
            with gr.Tab("Target Objects"):
                target_entity_id = gr.Textbox(label="Resolved entity ID")
                target_type = gr.Textbox(
                    label="Target object type",
                    value="target-system-object",
                )
                target_fields = gr.Textbox(
                    label="Fields to populate (comma-separated)",
                    value="country,status",
                )
                target_button = gr.Button("Create cited draft")
                target_status = gr.Markdown()
                target_result = gr.JSON(label="Analyst-review draft")
                target_button.click(
                    fn=lambda entity_id, object_type, fields: _draft_target_from_ui(
                        settings,
                        entity_id,
                        object_type,
                        fields,
                    ),
                    inputs=[target_entity_id, target_type, target_fields],
                    outputs=[target_status, target_result],
                )
                gr.Markdown("### Analyst decision")
                target_object_id = gr.Textbox(label="Draft object ID")
                target_decision = gr.Dropdown(
                    choices=[
                        ReviewDecision.APPROVED.value,
                        ReviewDecision.REJECTED.value,
                    ],
                    value=ReviewDecision.APPROVED.value,
                    label="Decision",
                )
                target_note = gr.Textbox(label="Analyst review note", lines=2)
                target_decide_button = gr.Button("Record analyst decision")
                target_decide_button.click(
                    fn=lambda object_id, decision, note: _decide_target_from_ui(
                        settings,
                        object_id,
                        decision,
                        note,
                    ),
                    inputs=[target_object_id, target_decision, target_note],
                    outputs=[target_status, target_result],
                )
    return demo
