import { mockAnalysis, mockSources } from '../mockData'
import { sampleDocument } from '../mockData'
import type {
  AcquisitionMetadata, AnalysisResponse, ApiResult, ChangeEvent, DemoDocument,
  EntityCandidate, IngestionStatus, IntelligenceEntity, IntelligenceQuery,
  IntelligenceResponse, NativeReport, Relationship, RiskLevel, SourceRecord,
  ReviewDecision,
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''
const REQUEST_TIMEOUT = 150000
export const demoModeEnabled = import.meta.env.VITE_ENABLE_DEMO_MODE === 'true'
const DEMO_PASSWORD = import.meta.env.VITE_DEMO_PASSWORD
let accessToken: string | null = null
let demoAuthorized = false

interface BackendCitation { title: string; url: string | null; section?: string; verification_status?: string; excerpt?: string }
interface BackendFinding {
  id: string; category: string; title: string; severity: RiskLevel; confidence: number | null; grounding_status?: string
  excerpt: string; explanation: string; recommendation: string; citations: BackendCitation[]
}
interface BackendAnalysis {
  document_summary: string; overall_risk: RiskLevel; confidence: number | null; findings: BackendFinding[]; report?: NativeReport
  disclaimer?: string; engine?: string
}
interface BackendSource { id: string; title: string; organization?: string; authority: string; category: string; url: string }

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) { super(message); this.name = 'ApiError' }
}

export async function request<T>(path: string, init?: RequestInit, protectedRoute = false): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT)
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...(protectedRoute && accessToken ? { Authorization: `Bearer ${accessToken}` } : {}),
        ...init?.headers,
      },
    })
    if (!response.ok) {
      let detail: unknown = ''
      try { detail = ((await response.json()) as { detail?: unknown }).detail ?? '' } catch { /* response may not be JSON */ }
      const safeDetail = typeof detail === 'string' ? detail : ''
      throw new ApiError(response.status === 401 ? 'That password was not accepted, or your session has expired.' : safeDetail || 'The service could not complete this request.', response.status)
    }
    return await response.json() as T
  } catch (error) {
    if (error instanceof ApiError) throw error
    throw new ApiError(error instanceof DOMException && error.name === 'AbortError' ? 'The local API timed out. Confirm it is running and try again.' : 'The local API is unavailable. Confirm it is running and try again.')
  } finally { window.clearTimeout(timeout) }
}

function mayUseDemo(password?: string) {
  return demoModeEnabled && Boolean(DEMO_PASSWORD) && password === DEMO_PASSWORD
}

function requireLiveIntelligence() {
  if (demoAuthorized) {
    throw new ApiError('This J2 workflow requires the live API. Offline demo mode does not generate or simulate intelligence results.')
  }
}

function mapAnalysis(response: BackendAnalysis): AnalysisResponse {
  return {
    documentSummary: response.document_summary,
    overallRisk: response.overall_risk,
    confidence: response.confidence,
    report: response.report,
    analyzedAt: response.engine ? `Engine: ${response.engine}` : undefined,
    disclaimer: response.disclaimer,
    findings: response.findings.map((finding) => ({
      id: finding.id, category: finding.category, title: finding.title, risk: finding.severity,
      confidence: finding.confidence, excerpt: finding.excerpt, explanation: finding.explanation,
      groundingStatus: finding.grounding_status,
      recommendation: finding.recommendation,
      sources: finding.citations.map((citation) => ({
        title: citation.title, url: citation.url && /^https?:\/\//i.test(citation.url) ? citation.url : '', citation: citation.section,
        excerpt: citation.excerpt,
        verificationStatus: citation.verification_status,
      })),
    })),
  }
}

export const apiClient = {
  async uploadFile(file: File, metadata: AcquisitionMetadata): Promise<ApiResult<AnalysisResponse>> {
    const body = new FormData()
    body.append('file', file)
    body.append('metadata_json', JSON.stringify(metadata))
    const response = await request<BackendAnalysis>('/api/review-upload', { method: 'POST', body }, true)
    return { data: mapAnalysis(response), demoMode: false }
  },
  async login(password: string): Promise<ApiResult<{ authenticated: boolean }>> {
    accessToken = null
    demoAuthorized = false
    try {
      const data = await request<{ access_token: string; token_type: string }>('/api/auth/login', {
        method: 'POST', body: JSON.stringify({ password }),
      })
      accessToken = data.access_token
      demoAuthorized = false
      return { data: { authenticated: Boolean(accessToken) }, demoMode: false }
    } catch (error) {
      if (error instanceof ApiError && error.status === undefined && mayUseDemo(password)) {
        accessToken = null
        demoAuthorized = true
        await new Promise((resolve) => window.setTimeout(resolve, 350))
        return { data: { authenticated: true }, demoMode: true }
      }
      throw error
    }
  },
  logout() {
    accessToken = null
    demoAuthorized = false
  },
  async getSample(): Promise<ApiResult<DemoDocument>> {
    try {
      const response = await request<DemoDocument>('/api/demo/sample', undefined, true)
      return { data: response, demoMode: false }
    } catch (error) {
      if (demoAuthorized) {
        return {
          data: { title: 'Technology Services Solicitation — Section I', text: sampleDocument },
          demoMode: true,
        }
      }
      throw error
    }
  },
  async getSources(): Promise<ApiResult<SourceRecord[]>> {
    try {
      const response = await request<BackendSource[]>('/api/sources', undefined, true)
      return { data: response.map((source) => ({
        id: source.id, title: source.title, organization: source.organization ?? source.authority,
        type: source.category, url: source.url,
        status: source.authority.toLowerCase() === 'official' ? 'Authoritative' : 'Reference',
      })), demoMode: false }
    } catch (error) {
      if (demoAuthorized) return { data: mockSources, demoMode: true }
      throw error
    }
  },
  async analyze(title: string, text: string, metadata?: AcquisitionMetadata): Promise<ApiResult<AnalysisResponse>> {
    try {
      const response = await request<BackendAnalysis>('/api/analyze', {
        method: 'POST', body: JSON.stringify({ title, text, metadata }),
      }, true)
      return { data: mapAnalysis(response), demoMode: false }
    } catch (error) {
      if (demoAuthorized) {
        if (text !== sampleDocument) {
          throw new ApiError('Offline presentation mode analyzes only the judge-ready sample. Load the sample and try again.')
        }
        await new Promise((resolve) => window.setTimeout(resolve, 900))
        return { data: mockAnalysis, demoMode: true }
      }
      throw error
    }
  },
  async queryIntelligence(payload: IntelligenceQuery): Promise<IntelligenceResponse> {
    requireLiveIntelligence()
    return request('/api/v1/intelligence/query', {
      method: 'POST', body: JSON.stringify(payload),
    }, true)
  },
  async getIngestionStatus(): Promise<IngestionStatus> {
    requireLiveIntelligence()
    return request('/api/v1/intelligence/ingestion/status', undefined, true)
  },
  async getEntities(): Promise<IntelligenceEntity[]> {
    requireLiveIntelligence()
    return request('/api/v1/intelligence/entities', undefined, true)
  },
  async resolveEntities(candidates: EntityCandidate[]): Promise<IntelligenceEntity[]> {
    requireLiveIntelligence()
    return request('/api/v1/intelligence/entities/resolve', {
      method: 'POST', body: JSON.stringify({ candidates }),
    }, true)
  },
  async getChanges(): Promise<ChangeEvent[]> {
    requireLiveIntelligence()
    return request('/api/v1/intelligence/changes', undefined, true)
  },
  async getRelationships(): Promise<Relationship[]> {
    requireLiveIntelligence()
    return request('/api/v1/intelligence/relationships', undefined, true)
  },
  async decideEntity(entityId: string, decision: Exclude<ReviewDecision, 'draft'>, note: string): Promise<IntelligenceEntity> {
    requireLiveIntelligence()
    return request(`/api/v1/intelligence/entities/${encodeURIComponent(entityId)}/decision`, {
      method: 'POST', body: JSON.stringify({ decision, note }),
    }, true)
  },
}
