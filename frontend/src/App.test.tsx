import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const analysisPayload = {
  analysis_id: 'test-analysis', title: 'Test',
  document_summary: 'A test solicitation with one clause that requires review.',
  overall_risk: 'high', confidence: 0.92, engine: 'deterministic-demo-v1',
  disclaimer: 'Demo screening output only—not legal advice.',
  findings: [{
    id: 'ip-rights', category: 'Data & IP rights', title: 'Unlimited rights require review',
    severity: 'high', confidence: 0.94, status: 'flagged', excerpt: '“unlimited rights”',
    explanation: 'The language may reach background intellectual property.',
    recommendation: 'Request a rights allocation review.',
    citations: [{ title: 'DFARS', url: 'https://www.acquisition.gov/dfars', section: 'Part 227', verification_status: 'official_source_applicability_unverified' }],
  }],
  graph: { nodes: [], edges: [] },
}

const backendSample = {
  metadata: { agency: 'Department of Defense', solicitation_number: 'DEMO-1', contract_type: 'firm-fixed-price', estimated_value: 1500000, set_aside: 'Small Business', commercial_product: false, cots_only: false, performance_months: 12, place_of_performance: 'Virginia', acquisition_stage: 'solicitation' },
  title: 'Sentinel Data Services — Draft Solicitation',
  text: `The Government owns all right, title, and interest in all data and software used during performance.
  FAR 52.249-2 Termination for Convenience applies to this fictional sample document.`,
}

function jsonResponse(body: unknown) {
  return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }))
}

describe('Acquisition Lens workflow', () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('authenticates, sends a bearer token, and produces source-backed findings', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input)
      if (url.endsWith('/api/auth/login')) return jsonResponse({ access_token: 'opaque-test-token', token_type: 'bearer', expires_in: 14400 })
      if (url.endsWith('/api/demo/sample')) return jsonResponse(backendSample)
      if (url.endsWith('/api/analyze')) return jsonResponse(analysisPayload)
      return jsonResponse([])
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText(/workspace password/i), 'safe-password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    expect(await screen.findByRole('heading', { name: /find the clause/i })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /load judge-ready sample/i }))
    expect(await screen.findByDisplayValue(backendSample.title)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /analyze contract/i }))

    expect(await screen.findByRole('heading', { name: /analysis results/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /unlimited rights require review/i })).toBeInTheDocument()
    expect(screen.getByText(/human review required/i)).toBeInTheDocument()
    expect(screen.getByText(/demo screening output only/i)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /analysis results/i })).toHaveFocus()
    const sampleCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/api/demo/sample'))
    expect((sampleCall?.[1]?.headers as Record<string, string>).Authorization).toBe('Bearer opaque-test-token')
    const analyzeCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/api/analyze'))
    expect((analyzeCall?.[1]?.headers as Record<string, string>).Authorization).toBe('Bearer opaque-test-token')
    expect(JSON.parse(analyzeCall?.[1]?.body as string).metadata).toEqual(backendSample.metadata)
  })

  it('rejects short submissions with an accessible error', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(() => jsonResponse({ access_token: 'token', token_type: 'bearer' }))
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText(/workspace password/i), 'safe-password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    await waitFor(() => expect(screen.getByRole('heading', { name: /find the clause/i })).toBeInTheDocument())
    await user.type(screen.getByLabelText(/contract or solicitation text/i), 'Too short')
    await user.click(screen.getByRole('button', { name: /analyze contract/i }))
    expect(screen.getByRole('alert')).toHaveTextContent(/at least 80 characters/i)
  })

  it('fails closed when the backend cannot authenticate', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'))
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText(/workspace password/i), 'any-password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/local api is unavailable/i)
    expect(screen.queryByRole('heading', { name: /find the clause/i })).not.toBeInTheDocument()
  })

  it('maps and authorizes the backend source catalog', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input)
      if (url.endsWith('/api/auth/login')) return jsonResponse({ access_token: 'source-token', token_type: 'bearer' })
      if (url.endsWith('/api/sources')) return jsonResponse([{
        id: 'far-part-52',
        title: 'FAR Part 52 — Solicitation Provisions and Contract Clauses',
        authority: 'Official',
        category: 'Federal rules and clauses',
        description: 'Official clauses.',
        url: 'https://www.acquisition.gov/far/part-52',
      }])
      return jsonResponse([])
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText(/workspace password/i), 'safe-password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    await user.click(await screen.findByRole('button', { name: /source library/i }))

    expect(await screen.findByRole('heading', { name: /far part 52/i })).toBeInTheDocument()
    expect(screen.getByText('Authoritative')).toBeInTheDocument()
    const sourceCall = fetchMock.mock.calls.find(([input]) => String(input).endsWith('/api/sources'))
    expect((sourceCall?.[1]?.headers as Record<string, string>).Authorization).toBe('Bearer source-token')
  })

  it('renders hostile analysis strings as inert text', async () => {
    const hostile = '<img src=x onerror="globalThis.__qaExecuted=true"><script>globalThis.__qaExecuted=true</script>'
    const payload = {
      ...analysisPayload,
      findings: [{
        ...analysisPayload.findings[0],
        title: hostile,
        excerpt: hostile,
        explanation: hostile,
        recommendation: hostile,
      }],
    }
    vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const url = String(input)
      if (url.endsWith('/api/auth/login')) return jsonResponse({ access_token: 'safe-token', token_type: 'bearer' })
      if (url.endsWith('/api/analyze')) return jsonResponse(payload)
      return jsonResponse([])
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(screen.getByLabelText(/workspace password/i), 'safe-password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    await user.type(await screen.findByLabelText(/contract or solicitation text/i), 'x'.repeat(100))
    await user.type(screen.getByLabelText(/^agency$/i), 'Department of Defense')
    await user.type(screen.getByLabelText(/^solicitation number$/i), 'DEMO-1')
    await user.type(screen.getByLabelText(/^place of performance$/i), 'Virginia')
    await user.click(screen.getByRole('button', { name: /analyze contract/i }))

    expect(await screen.findByRole('heading', { name: hostile })).toBeInTheDocument()
    expect(screen.getAllByText(hostile).length).toBeGreaterThanOrEqual(4)
    expect(document.querySelector('img[src="x"]')).toBeNull()
    expect(document.querySelector('.finding-card script')).toBeNull()
    expect((globalThis as typeof globalThis & { __qaExecuted?: boolean }).__qaExecuted).toBeUndefined()
    expect(localStorage).toHaveLength(0)
    expect(sessionStorage).toHaveLength(0)
  })
})
