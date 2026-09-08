export type RiskLevel = 'critical' | 'high' | 'medium' | 'low' | 'info'

export interface AcquisitionMetadata {
  agency: string
  solicitation_number: string
  contract_type: string
  estimated_value: number
  set_aside: string
  commercial_product: boolean
  cots_only: boolean
  performance_months: number
  place_of_performance: string
  acquisition_stage: string
}

export const emptyMetadata: AcquisitionMetadata = {
  agency: '', solicitation_number: '', contract_type: 'firm-fixed-price',
  estimated_value: 0, set_aside: 'None', commercial_product: false,
  cots_only: false, performance_months: 12, place_of_performance: '',
  acquisition_stage: 'solicitation',
}

export interface NativeReport {
  corpus_manifest?: {
    documents?: number
    chunks?: number
    links?: number
    retrieved_at?: string
  }
  knowledge_graph?: { source_references?: { source: string; target: string; relation: string }[] }
  report_id: string
  synthesis_mode: string
  classifier_model_ids: string[]
  clause_status_inventory: {
    clause_id: string; title: string; status: string; rationale: string
  }[]
}

export interface SourceCitation {
  title: string
  url: string
  citation?: string
  verificationStatus?: string
  excerpt?: string
}

export interface Finding {
  id: string
  category: string
  title: string
  risk: RiskLevel
  confidence: number | null
  groundingStatus?: string
  excerpt: string
  explanation: string
  recommendation: string
  sources: SourceCitation[]
}

export interface AnalysisResponse {
  documentSummary: string
  overallRisk: RiskLevel
  confidence: number | null
  report?: NativeReport
  findings: Finding[]
  analyzedAt?: string
  disclaimer?: string
}

export interface SourceRecord {
  id: string
  title: string
  organization: string
  type: string
  url: string
  status: 'Authoritative' | 'Reference'
}

export interface ApiResult<T> {
  data: T
  demoMode: boolean
}

export interface DemoDocument {
  title: string
  text: string
  metadata?: AcquisitionMetadata
}

export type GenerationMode = 'answer' | 'summary' | 'draft'
export type GroundingStatus = 'verified' | 'unverified'
export type ReviewDecision = 'draft' | 'approved' | 'rejected'
export type EntityAttribute = string | number | boolean | null

export interface CorpusFilters {
  source_types: string[]
  document_ids: string[]
  entity_ids: string[]
  effective_after: string | null
}

export interface IntelligenceQuery {
  query: string
  mode: GenerationMode
  workflow: string
  filters: CorpusFilters
}

export interface ProvenanceLink {
  citation_id: string
  document_id: string
  version: string
  content_sha256: string
}

export interface IntelligenceEvidence {
  evidence_id: string
  source: string
  title: string
  excerpt: string
  url: string | null
  document_id: string
  version: string
  security_label: string
  acl_principals: string[]
  entity_ids: string[]
}

export interface CitedStatement {
  text: string
  citation_ids: string[]
  grounding_status: GroundingStatus
}

export interface IntelligenceResponse {
  response_id: string
  generated_at: string
  query: string
  mode: GenerationMode
  workflow: string
  answer: string
  statements: CitedStatement[]
  evidence: IntelligenceEvidence[]
  synthesis_mode: string
}

export interface IngestionStatus {
  security_domain: string
  fixture_documents_expected: number
  durable_documents: number | null
  durable_store_configured: boolean
  graph_connector_configured: boolean
}

export interface EntityCandidate {
  name: string
  entity_type: string
  attributes: Record<string, EntityAttribute>
  aliases: string[]
  relationships: string[]
  provenance: ProvenanceLink[]
}

export interface IntelligenceEntity {
  entity_id: string
  canonical_name: string
  normalized_name: string
  entity_type: string
  attributes: Record<string, EntityAttribute>
  aliases: string[]
  relationship_ids: string[]
  provenance: ProvenanceLink[]
  review_status: ReviewDecision
  reviewed_by: string | null
  review_note: string
  updated_at: string
}

export interface ChangeEvent {
  change_id: string
  entity_id: string
  changed_fields: string[]
  before: Record<string, unknown>
  after: Record<string, unknown>
  provenance: ProvenanceLink[]
  detected_at: string
}

export interface Relationship {
  relationship_id: string
  source_entity_id: string
  relationship_type: string
  target_entity_id: string
  citation_ids: string[]
  grounding_status: GroundingStatus
}
