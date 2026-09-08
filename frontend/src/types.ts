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
  corpus_manifest?: { documents: number; chunks: number; links: number; retrieved_at: string }
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
