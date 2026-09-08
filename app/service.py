"""Contract review orchestration and trust-boundary validation."""

from collections.abc import Iterable

from app.adapters import (
    BedrockGPT56TerraSynthesisAdapter,
    ClauseClassificationAdapter,
    FallbackRetrievalAdapter,
    KeywordClauseClassificationAdapter,
    LocalRetrievalAdapter,
    OpenSearchRetrievalAdapter,
    PackagedMLClassificationAdapter,
    RetrievalAdapter,
    RuleEngineAdapter,
    SynthesisAdapter,
    YamlRuleEngineAdapter,
    dfars_7012_applicable,
)
from app.config import Settings
from app.grounding import authorized_citation_ids
from app.models import (
    AblationSummary,
    AcquisitionMetadata,
    ClassifiedClause,
    ClauseStatus,
    ClauseStatusItem,
    FindingContext,
    GroundingStatus,
    ParsedDocument,
    PrincipalContext,
    ReviewReport,
    RuleFinding,
    Severity,
)


class ReviewService:
    def __init__(
        self,
        settings: Settings,
        classifier: ClauseClassificationAdapter | None = None,
        rules: RuleEngineAdapter | None = None,
        retrieval: RetrievalAdapter | None = None,
        synthesis: SynthesisAdapter | None = None,
    ) -> None:
        self.classifier = classifier or PackagedMLClassificationAdapter(settings)
        self.rules = rules or YamlRuleEngineAdapter(settings)
        self.retrieval = retrieval or self._build_retrieval(settings)
        self.synthesis = synthesis or BedrockGPT56TerraSynthesisAdapter(settings)

    @staticmethod
    def _build_retrieval(settings: Settings) -> RetrievalAdapter:
        if settings.local_corpus_path:
            from app.local_retrieval import FederalCorpusRetrieval

            return FederalCorpusRetrieval(settings.local_corpus_path)
        local = LocalRetrievalAdapter()
        if not settings.opensearch_endpoint:
            return local
        return FallbackRetrievalAdapter(
            OpenSearchRetrievalAdapter(settings),
            fallback=local,
        )

    def review(self, document: ParsedDocument, metadata: AcquisitionMetadata) -> ReviewReport:
        deterministic_clauses = KeywordClauseClassificationAdapter().classify(document)
        clauses = self.classifier.classify(document)
        deterministic_findings = self.rules.evaluate(
            document,
            metadata,
            deterministic_clauses,
        )
        rule_findings = self.rules.evaluate(document, metadata, clauses)
        evidence_by_id = {}
        allowed_citations_by_rule: dict[str, set[str]] = {}
        for rule_finding in rule_findings:
            rule_evidence = self.retrieval.retrieve(rule_finding.evidence_queries)
            allowed_citations_by_rule[rule_finding.rule_id] = {
                item.evidence_id for item in rule_evidence
            }
            for item in rule_evidence:
                evidence_by_id.setdefault(item.evidence_id, item)
        evidence = list(evidence_by_id.values())
        review_principal = PrincipalContext(
            subject="contract-review",
            groups=["contract-reviewers"],
            security_domain="demo",
        )
        authorized_ids = authorized_citation_ids(evidence, review_principal)
        summary, findings, mode = self.synthesis.synthesize(rule_findings, evidence)

        locations = {segment.segment_id: segment.location for segment in document.segments}
        source_findings = {item.rule_id: item for item in rule_findings}
        for finding in findings:
            allowed_citations = (
                allowed_citations_by_rule.get(finding.rule_id, set()) & authorized_ids
            )
            finding.citation_ids = list(
                self._stable_unique(
                    citation for citation in finding.citation_ids if citation in allowed_citations
                )
            )
            if finding.citation_ids:
                finding.grounding_status = GroundingStatus.VERIFIED
            else:
                finding.severity = Severity.INFO
                finding.grounding_status = GroundingStatus.UNVERIFIED
                if not finding.description.startswith("UNVERIFIED —"):
                    finding.description = (
                        "UNVERIFIED — no retrieved evidence supports this synthesized "
                        f"finding. {finding.description}"
                    )
            source = source_findings.get(finding.rule_id)
            if source:
                finding.document_context = [
                    FindingContext(
                        text=next(
                            segment.text
                            for segment in document.segments
                            if segment.segment_id == segment_id
                        ),
                        location=locations[segment_id],
                    )
                    for segment_id in source.segment_ids
                    if segment_id in locations
                ]
                finding.locations = [context.location for context in finding.document_context]
                related_clauses = [
                    clause for clause in clauses if clause.segment_id in source.segment_ids
                ]
                if related_clauses:
                    finding.classifier_confidence = max(
                        clause.confidence for clause in related_clauses
                    )
                    finding.classifier_model_ids = list(
                        self._stable_unique(
                            clause.classifier_model_id for clause in related_clauses
                        )
                    )

        unverified_count = sum(
            finding.grounding_status == GroundingStatus.UNVERIFIED for finding in findings
        )
        if unverified_count:
            high_count = sum(
                finding.severity in {Severity.CRITICAL, Severity.HIGH} for finding in findings
            )
            summary = (
                f"Review identified {len(findings)} potential issue(s) after citation "
                f"validation, including {high_count} high or critical item(s). "
                f"{unverified_count} unsupported item(s) were marked unverified and "
                "downgraded to informational."
            )

        deterministic_keys = {
            (clause.clause_id, clause.segment_id) for clause in deterministic_clauses
        }
        assisted_keys = {(clause.clause_id, clause.segment_id) for clause in clauses}
        classifier_model_ids = list(
            self._stable_unique(
                [
                    *getattr(self.classifier, "model_ids", []),
                    *(clause.classifier_model_id for clause in clauses),
                ]
            )
        )
        return ReviewReport(
            filename=document.filename,
            document_sha256=document.sha256,
            metadata=metadata,
            executive_summary=summary,
            findings=findings,
            clause_inventory=clauses,
            clause_status_inventory=self._build_clause_status_inventory(
                document,
                metadata,
                clauses,
                rule_findings,
            ),
            evidence=evidence,
            synthesis_mode=mode,
            classifier_model_ids=classifier_model_ids,
            ablation_summary=AblationSummary(
                deterministic_clause_count=len(deterministic_keys),
                classifier_assisted_clause_count=len(assisted_keys),
                classifier_added_clause_count=len(assisted_keys - deterministic_keys),
                deterministic_finding_count=len(deterministic_findings),
                classifier_assisted_finding_count=len(rule_findings),
                retrieved_evidence_count=len(evidence),
            ),
        )

    @staticmethod
    def _build_clause_status_inventory(
        document: ParsedDocument,
        metadata: AcquisitionMetadata,
        clauses: list[ClassifiedClause],
        rule_findings: list[RuleFinding],
    ) -> list[ClauseStatusItem]:
        grouped: dict[str, list[ClassifiedClause]] = {}
        for clause in clauses:
            grouped.setdefault(clause.clause_id, []).append(clause)
        inventory = [
            ClauseStatusItem(
                clause_id=clause_id,
                title=detected[0].title,
                category=detected[0].category,
                status=(
                    ClauseStatus.TRIAGE_CANDIDATE
                    if clause_id.startswith("TRIAGE:")
                    else ClauseStatus.PRESENT
                ),
                rationale=(
                    "Classifier triage evidence with positive structural language was "
                    "detected; this does not establish clause presence or applicability."
                    if clause_id.startswith("TRIAGE:")
                    else "Clause text was detected in the uploaded document."
                ),
                detected_locations=[item.location for item in detected],
                confidence=max(item.confidence for item in detected),
                classifier_model_ids=list(
                    dict.fromkeys(item.classifier_model_id for item in detected)
                ),
            )
            for clause_id, detected in grouped.items()
        ]

        missing_rules = {
            "GC-004": (
                "DFARS 252.204-7012",
                "Safeguarding Covered Defense Information",
                "cybersecurity",
            ),
            "sam-registration-far-52-204-7": (
                "FAR 52.204-7",
                "System for Award Management",
                "registration",
            ),
        }
        existing_ids = set(grouped)
        segments = {segment.segment_id: segment.location for segment in document.segments}
        for finding in rule_findings:
            clause_details = missing_rules.get(finding.rule_id)
            if clause_details and clause_details[0] not in existing_ids:
                clause_id, title, category = clause_details
                inventory.append(
                    ClauseStatusItem(
                        clause_id=clause_id,
                        title=title,
                        category=category,
                        status=ClauseStatus.MISSING,
                        rationale=(
                            f"{finding.description} Applicability is based only on "
                            "configured deterministic review rules and requires human confirmation."
                        ),
                        detected_locations=[
                            segments[item] for item in finding.segment_ids if item in segments
                        ],
                    )
                )
                existing_ids.add(clause_id)
            if finding.rule_id == "malformed-clause-reference":
                inventory.append(
                    ClauseStatusItem(
                        clause_id="UNRESOLVED-CLAUSE-REFERENCE",
                        title="Malformed FAR/DFARS reference",
                        category="formatting",
                        status=ClauseStatus.MALFORMED,
                        rationale=finding.description,
                        detected_locations=[
                            segments[item] for item in finding.segment_ids if item in segments
                        ],
                    )
                )

        cybersecurity_id = "DFARS 252.204-7012"
        if cybersecurity_id not in existing_ids:
            if metadata.cots_only:
                cybersecurity_status = ClauseStatus.NOT_APPLICABLE
                cybersecurity_rationale = (
                    "Metadata identifies the acquisition as solely COTS; this DFARS "
                    "inventory check is not applied."
                )
            elif dfars_7012_applicable(metadata):
                cybersecurity_status = ClauseStatus.MISSING
                cybersecurity_rationale = (
                    "A DoD solicitation/contract-stage review found no clause identifier. "
                    "Confirm covered defense information applicability with counsel."
                )
            else:
                cybersecurity_status = ClauseStatus.UNDETERMINED
                cybersecurity_rationale = (
                    "Metadata does not establish the limited DoD, contract-stage, "
                    "non-COTS conditions used by this demo. No legal applicability "
                    "determination is made."
                )
            inventory.append(
                ClauseStatusItem(
                    clause_id=cybersecurity_id,
                    title="Safeguarding Covered Defense Information",
                    category="cybersecurity",
                    status=cybersecurity_status,
                    rationale=cybersecurity_rationale,
                )
            )
            existing_ids.add(cybersecurity_id)

        commercial_id = "FAR 52.212-4"
        if commercial_id not in existing_ids:
            if metadata.commercial_product:
                status = ClauseStatus.UNDETERMINED
                rationale = (
                    "Metadata identifies a commercial product/service and the clause was "
                    "not detected, but this demo does not determine legal applicability."
                )
            else:
                status = ClauseStatus.NOT_APPLICABLE
                rationale = (
                    "Commercial product/service metadata is false, so this specialized "
                    "inventory check is not applied."
                )
            inventory.append(
                ClauseStatusItem(
                    clause_id=commercial_id,
                    title="Contract Terms and Conditions—Commercial Products and Services",
                    category="commercial",
                    status=status,
                    rationale=rationale,
                )
            )
        return inventory

    @staticmethod
    def _stable_unique(values: Iterable[str]) -> Iterable[str]:
        seen: set[str] = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                yield value
