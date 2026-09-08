"""Typed domain models shared by API, UI, and adapters."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class ContractType(StrEnum):
    FIRM_FIXED_PRICE = "firm-fixed-price"
    COST_REIMBURSEMENT = "cost-reimbursement"
    TIME_AND_MATERIALS = "time-and-materials"
    INDEFINITE_DELIVERY = "indefinite-delivery"
    OTHER = "other"


class AcquisitionStage(StrEnum):
    PRE_SOLICITATION = "pre-solicitation"
    SOLICITATION = "solicitation"
    EVALUATION = "evaluation"
    AWARD = "award"
    POST_AWARD = "post-award"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class GroundingStatus(StrEnum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class GenerationMode(StrEnum):
    ANSWER = "answer"
    SUMMARY = "summary"
    DRAFT = "draft"


class ReviewDecision(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REJECTED = "rejected"


class ClauseStatus(StrEnum):
    PRESENT = "present"
    TRIAGE_CANDIDATE = "triage-candidate"
    MISSING = "missing"
    MALFORMED = "malformed"
    NOT_APPLICABLE = "not-applicable"
    UNDETERMINED = "undetermined"


class AcquisitionMetadata(BaseModel):
    agency: str = Field(min_length=1, max_length=200)
    solicitation_number: str = Field(min_length=1, max_length=100)
    contract_type: ContractType = ContractType.FIRM_FIXED_PRICE
    estimated_value: float = Field(default=0, ge=0)
    set_aside: str = Field(default="None", max_length=100)
    commercial_product: bool = False
    cots_only: bool = False
    performance_months: int = Field(default=12, ge=1, le=240)
    place_of_performance: str = Field(min_length=1, max_length=300)
    acquisition_stage: AcquisitionStage

    @model_validator(mode="after")
    def validate_cots_is_commercial(self) -> "AcquisitionMetadata":
        if self.cots_only and not self.commercial_product:
            raise ValueError("cots_only requires commercial_product")
        return self


class DocumentLocation(BaseModel):
    page: int | None = Field(default=None, ge=1)
    paragraph: int = Field(ge=1)
    label: str


class ExtractedSegment(BaseModel):
    segment_id: str
    text: str
    location: DocumentLocation


class ParsedDocument(BaseModel):
    filename: str
    media_type: str
    sha256: str
    segments: list[ExtractedSegment]

    @property
    def text(self) -> str:
        return "\n\n".join(segment.text for segment in self.segments)


class ClassifiedClause(BaseModel):
    clause_id: str
    title: str
    category: str
    confidence: float = Field(ge=0, le=1)
    segment_id: str
    location: DocumentLocation
    excerpt: str
    classifier_model_id: str = Field(min_length=1)


class ClauseStatusItem(BaseModel):
    clause_id: str
    title: str
    category: str
    status: ClauseStatus
    rationale: str
    detected_locations: list[DocumentLocation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    classifier_model_ids: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str
    source: str
    title: str
    excerpt: str
    url: str | None = None
    document_id: str = ""
    version: str = ""
    security_label: str = "public"
    acl_principals: list[str] = Field(default_factory=lambda: ["public"])
    entity_ids: list[str] = Field(default_factory=list)


class PrincipalContext(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    groups: list[str] = Field(default_factory=list)
    security_domain: str = Field(default="demo", min_length=1, max_length=40)
    scopes: list[str] = Field(default_factory=list)

    @property
    def acl_principals(self) -> set[str]:
        return {
            "public",
            f"user:{self.subject}",
            *(f"group:{group}" for group in self.groups),
        }


class CorpusFilters(BaseModel):
    source_types: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    effective_after: str | None = None


class IntelligenceQuery(BaseModel):
    query: str = Field(min_length=2, max_length=4_000)
    mode: GenerationMode = GenerationMode.ANSWER
    workflow: str = Field(default="mission-support", min_length=1, max_length=100)
    filters: CorpusFilters = Field(default_factory=CorpusFilters)


class CitedStatement(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    grounding_status: GroundingStatus = GroundingStatus.UNVERIFIED


class IntelligenceResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query: str
    mode: GenerationMode
    workflow: str
    answer: str
    statements: list[CitedStatement]
    evidence: list[Evidence]
    synthesis_mode: str


class ProvenanceLink(BaseModel):
    citation_id: str
    document_id: str
    version: str = ""
    content_sha256: str = ""


class EntityCandidate(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    entity_type: str = Field(min_length=1, max_length=100)
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceLink] = Field(default_factory=list)


class IntelligenceEntity(BaseModel):
    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    canonical_name: str
    normalized_name: str
    entity_type: str
    attributes: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    relationship_ids: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceLink] = Field(default_factory=list)
    review_status: ReviewDecision = ReviewDecision.DRAFT
    reviewed_by: str | None = None
    review_note: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EntityResolutionRequest(BaseModel):
    candidates: list[EntityCandidate] = Field(min_length=1, max_length=500)


class ChangeEvent(BaseModel):
    change_id: str = Field(default_factory=lambda: str(uuid4()))
    entity_id: str
    changed_fields: list[str]
    before: dict[str, object] = Field(default_factory=dict)
    after: dict[str, object] = Field(default_factory=dict)
    provenance: list[ProvenanceLink] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Relationship(BaseModel):
    relationship_id: str = Field(default_factory=lambda: str(uuid4()))
    source_entity_id: str
    relationship_type: str
    target_entity_id: str
    citation_ids: list[str] = Field(default_factory=list)
    grounding_status: GroundingStatus = GroundingStatus.UNVERIFIED


class AnalystDecisionRequest(BaseModel):
    decision: ReviewDecision
    note: str = Field(default="", max_length=2_000)


class RuleFinding(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    description: str
    recommendation: str
    segment_ids: list[str] = Field(default_factory=list)
    evidence_queries: list[str] = Field(default_factory=list)


class FindingContext(BaseModel):
    text: str
    location: DocumentLocation


class Finding(BaseModel):
    finding_id: str
    rule_id: str
    title: str
    severity: Severity
    description: str
    recommendation: str
    locations: list[DocumentLocation] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    grounding_status: GroundingStatus = GroundingStatus.UNVERIFIED
    document_context: list[FindingContext] = Field(default_factory=list)
    classifier_confidence: float | None = Field(default=None, ge=0, le=1)
    classifier_model_ids: list[str] = Field(default_factory=list)


class AblationSummary(BaseModel):
    deterministic_clause_count: int = Field(ge=0)
    classifier_assisted_clause_count: int = Field(ge=0)
    classifier_added_clause_count: int = Field(ge=0)
    deterministic_finding_count: int = Field(ge=0)
    classifier_assisted_finding_count: int = Field(ge=0)
    retrieved_evidence_count: int = Field(ge=0)


class ReviewReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    filename: str
    document_sha256: str
    metadata: AcquisitionMetadata
    executive_summary: str
    findings: list[Finding]
    clause_inventory: list[ClassifiedClause]
    clause_status_inventory: list[ClauseStatusItem]
    evidence: list[Evidence]
    synthesis_mode: str
    classifier_model_ids: list[str]
    ablation_summary: AblationSummary
    corpus_manifest: dict = Field(default_factory=dict)
    knowledge_graph: dict = Field(default_factory=dict)

    def severity_counts(self) -> dict[str, int]:
        return {
            severity.value: sum(finding.severity == severity for finding in self.findings)
            for severity in Severity
        }
