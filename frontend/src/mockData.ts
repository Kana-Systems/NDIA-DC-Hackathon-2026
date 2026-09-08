import type { AnalysisResponse, SourceRecord } from './types'

export const sampleDocument = `SECTION I — CONTRACT CLAUSES

52.204-21 Basic Safeguarding of Covered Contractor Information Systems (NOV 2021) is incorporated by reference.

The Contractor shall provide unlimited rights in all technical data, computer software, and documentation first produced or used in performance of this contract, including privately developed materials.

The Government may terminate this contract for convenience. In the event of termination, no payment will be made for work performed, accepted deliverables, or reasonable termination costs.

The Contractor warrants that all deliverables will be free from defects for a period of five (5) years following final acceptance and will correct any defect at no additional cost.

The Contractor shall notify the Contracting Officer of any cyber incident within 24 hours and preserve all relevant system images for 90 days.`

export const mockAnalysis: AnalysisResponse = {
  documentSummary:
    'A federal technology services solicitation containing five reviewable clause families. The language creates elevated intellectual-property, termination, warranty, and cybersecurity obligations that merit acquisition and counsel review before submission.',
  overallRisk: 'high',
  confidence: 0.91,
  analyzedAt: 'Deterministic demo analysis',
  findings: [
    {
      id: 'ip-rights', category: 'Data & IP rights', title: 'Unlimited rights reach privately developed material', risk: 'critical', confidence: 0.96,
      excerpt: 'The Contractor shall provide unlimited rights in all technical data, computer software, and documentation first produced or used in performance of this contract, including privately developed materials.',
      explanation: 'This grant appears broader than standard rights allocations and may transfer valuable background IP beyond material developed under the award.',
      recommendation: 'Create a schedule of background IP and propose rights language that distinguishes pre-existing material from contract-funded deliverables.',
      sources: [{ title: 'DFARS', url: 'https://www.acquisition.gov/dfars', citation: 'Part 227 and applicable clauses — verify with counsel', verificationStatus: 'candidate_source' }],
    },
    {
      id: 'termination', category: 'Termination', title: 'Termination clause may waive customary cost recovery', risk: 'high', confidence: 0.93,
      excerpt: 'In the event of termination, no payment will be made for work performed, accepted deliverables, or reasonable termination costs.',
      explanation: 'The stated payment waiver is unusually broad and may conflict with the commercial and noncommercial termination frameworks normally incorporated into federal awards.',
      recommendation: 'Escalate before proposal submission and request alignment to the applicable FAR termination clause and cost-settlement process.',
      sources: [{ title: 'FAR Part 52', url: 'https://www.acquisition.gov/far/part-52', citation: 'Termination clause family — candidate source for human verification', verificationStatus: 'candidate_source' }],
    },
    {
      id: 'warranty', category: 'Warranty', title: 'Five-year no-cost warranty exceeds typical exposure', risk: 'medium', confidence: 0.84,
      excerpt: 'The Contractor warrants that all deliverables will be free from defects for a period of five (5) years following final acceptance and will correct any defect at no additional cost.',
      explanation: 'The warranty has a long duration, unlimited correction obligation, and no visible exclusions, acceptance criteria, or liability boundary.',
      recommendation: 'Define defect and acceptance criteria; negotiate a shorter period plus exclusions for misuse, third-party changes, and normal maintenance.',
      sources: [{ title: 'FAR Smart Matrix', url: 'https://www.acquisition.gov/smart-matrix', citation: 'Use matrix to confirm the prescribed warranty clause', verificationStatus: 'candidate_source' }],
    },
    {
      id: 'cyber', category: 'Cybersecurity', title: 'Incident reporting timeline requires process validation', risk: 'medium', confidence: 0.89,
      excerpt: 'The Contractor shall notify the Contracting Officer of any cyber incident within 24 hours and preserve all relevant system images for 90 days.',
      explanation: 'The requirement may be achievable, but the organization must confirm monitoring, escalation, evidence preservation, and subcontractor flow-down readiness.',
      recommendation: 'Map this duty to the incident-response plan and confirm the exact controlling cyber clause and reporting portal.',
      sources: [{ title: 'eCFR Title 48', url: 'https://www.ecfr.gov/current/title-48', citation: 'Cybersecurity provisions vary by agency — verify applicability', verificationStatus: 'candidate_source' }],
    },
  ],
}

export const mockSources: SourceRecord[] = [
  { id: 'far52', title: 'FAR Part 52', organization: 'Acquisition.gov', type: 'Clause library', url: 'https://www.acquisition.gov/far/part-52', status: 'Authoritative' },
  { id: 'far', title: 'Federal Acquisition Regulation', organization: 'Acquisition.gov', type: 'Regulation', url: 'https://www.acquisition.gov/browse/index/far', status: 'Authoritative' },
  { id: 'dfars', title: 'Defense FAR Supplement', organization: 'DoD', type: 'Regulation', url: 'https://www.acquisition.gov/dfars', status: 'Authoritative' },
  { id: 'ecfr', title: 'eCFR Title 48', organization: 'Office of the Federal Register', type: 'Regulation', url: 'https://www.ecfr.gov/current/title-48', status: 'Authoritative' },
  { id: 'gao', title: 'GAO Bid Protest Decisions', organization: 'U.S. GAO', type: 'Decision archive', url: 'https://www.gao.gov/legal/bid-protests/search', status: 'Reference' },
  { id: 'sam', title: 'SAM.gov Opportunities', organization: 'U.S. GSA', type: 'Market data', url: 'https://sam.gov/opportunities', status: 'Reference' },
]
