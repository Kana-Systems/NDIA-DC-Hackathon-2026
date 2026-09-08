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
})
