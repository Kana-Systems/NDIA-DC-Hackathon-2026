import { afterEach, describe, expect, it, vi } from 'vitest'


describe('explicit offline demo authorization', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    vi.resetModules()
  })

  it('does not replace a backend credential rejection with demo access', async () => {
    vi.stubEnv('VITE_ENABLE_DEMO_MODE', 'true')
    vi.stubEnv('VITE_DEMO_PASSWORD', 'offline-only')
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Invalid credentials' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    const { apiClient } = await import('./client')

    await expect(apiClient.login('offline-only')).rejects.toMatchObject({ status: 401 })
  })

  it('allows only the canonical sample after an offline demo login', async () => {
    vi.stubEnv('VITE_ENABLE_DEMO_MODE', 'true')
    vi.stubEnv('VITE_DEMO_PASSWORD', 'offline-only')
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'))
    const { apiClient } = await import('./client')
    const { sampleDocument } = await import('../mockData')

    await expect(apiClient.login('offline-only')).resolves.toMatchObject({
      data: { authenticated: true }, demoMode: true,
    })
    const sample = await apiClient.getSample()
    expect(sample.data.text).toBe(sampleDocument)
    await expect(apiClient.analyze('Custom', 'x'.repeat(100))).rejects.toThrow(/only the judge-ready sample/i)
    const analysis = await apiClient.analyze(sample.data.title, sample.data.text)
    expect(analysis.demoMode).toBe(true)
    expect(analysis.data.findings.every((finding) => sampleDocument.includes(finding.excerpt))).toBe(true)
  })

  it('uses the in-memory bearer token for every J2 workflow contract', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation((input) => {
      const path = String(input)
      if (path.endsWith('/api/auth/login')) {
        return Promise.resolve(new Response(JSON.stringify({ access_token: 'j2-token', token_type: 'bearer' }), { status: 200 }))
      }
      const body = path.endsWith('/ingestion/status')
        ? { security_domain: 'demo', fixture_documents_expected: 3, durable_documents: null, durable_store_configured: false, graph_connector_configured: false }
        : path.endsWith('/query')
          ? { response_id: 'r1', generated_at: '2026-01-01T00:00:00Z', query: 'question', mode: 'answer', workflow: 'mission-support', answer: 'answer', statements: [], evidence: [], synthesis_mode: 'extractive' }
          : path.includes('/entities/') || path.endsWith('/entities/resolve')
            ? { entity_id: 'e1' }
            : []
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }))
    })
    const { apiClient } = await import('./client')
    await apiClient.login('password')

    await apiClient.queryIntelligence({ query: 'question', mode: 'answer', workflow: 'mission-support', filters: { source_types: [], document_ids: [], entity_ids: [], effective_after: null } })
    await apiClient.getIngestionStatus()
    await apiClient.getEntities()
    await apiClient.resolveEntities([{ name: 'Entity', entity_type: 'system', attributes: {}, aliases: [], relationships: [], provenance: [] }])
    await apiClient.getChanges()
    await apiClient.getRelationships()
    await apiClient.decideEntity('entity/1', 'approved', 'Reviewed')

    const workflowCalls = fetchMock.mock.calls.filter(([input]) => String(input).includes('/api/v1/intelligence/'))
    expect(workflowCalls).toHaveLength(7)
    expect(workflowCalls.every(([, init]) => (init?.headers as Record<string, string>).Authorization === 'Bearer j2-token')).toBe(true)
    expect(workflowCalls.map(([input]) => String(input))).toContainEqual(expect.stringContaining('/entities/entity%2F1/decision'))
    expect(JSON.parse(workflowCalls[0][1]?.body as string)).toEqual({
      query: 'question', mode: 'answer', workflow: 'mission-support',
      filters: { source_types: [], document_ids: [], entity_ids: [], effective_after: null },
    })
  })
})
