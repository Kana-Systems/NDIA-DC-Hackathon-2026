import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { apiClient } from './api/client'

// Isolated UI fixtures, never included in deployed workspace data.
const doc = { id: 'document-1', kind: 'document', title: 'Uploaded agreement', text: 'The supplier warrants the original equipment for twelve months. '.repeat(3), category: 'contract', version: 'v1', metadata: null, revision: 1, owner: 'reviewer', created_at: '2026-09-08T12:00:00Z', updated_at: '2026-09-08T12:00:00Z', status: 'ready' }
const analysis = { document_summary: 'An uploaded agreement.', overall_risk: 'high', engine: 'test-engine', disclaimer: 'Human review required.', report: { classifier_model_ids: ['test-classifier'] }, findings: [{ id: 'warranty', title: 'Warranty requires review', category: 'Warranty', severity: 'high', excerpt: 'twelve months', explanation: 'Compare against the approved playbook.', recommendation: 'Confirm the required duration.', grounding_status: 'verified', citations: [{ title: 'Approved playbook', url: 'https://example.org/playbook', section: 'Warranty', excerpt: 'The required warranty duration.', verification_status: 'retrieved_evidence' }] }] }
function json(body: unknown) { return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })) }
function mockApi(documents: unknown[] = [], custom?: (url: string, init?: RequestInit) => ReturnType<typeof json> | undefined) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation((input, init) => {
    const url = String(input)
    const override = custom?.(url, init)
    if (override) return override
    if (url.endsWith('/api/auth/login')) return json({ access_token: 'test-token', token_type: 'bearer' })
    if (url.endsWith('/documents')) return json(documents)
    if (url.endsWith('/documents/document-1')) return json(doc)
    if (url.includes('/library?')) return json({ corpus: null, evidence: [], documents: [], catalog: [{ id: 'far', title: 'FAR Part 52', url: 'https://www.acquisition.gov/far/part-52', authority: 'Official', category: 'Federal rules' }] })
    if (url.endsWith('/connections')) return json({ connections: [], shared_folder_available: false, sharepoint_available: false, automatic_sync_seconds: 0, persistence: 'dynamodb', security_domain: 'demo' })
    return json([])
  })
}
async function login() {
  const user = userEvent.setup()
  render(<App />)
  await user.type(screen.getByLabelText(/workspace password/i), 'test-password')
  await user.click(screen.getByRole('button', { name: /enter workspace/i }))
  await screen.findByRole('heading', { name: /your contract workspace/i })
  return user
}
describe('Connected Lens workspace', () => {
  afterEach(() => { cleanup(); apiClient.logout(); vi.restoreAllMocks() })
  it('uses Kana Legal branding on sign-in and the workspace', async () => {
    mockApi()
    const user = userEvent.setup()
    render(<App />)
    expect(screen.getByText('Kana Legal')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Kana Systems' })).toHaveAttribute('src', expect.stringContaining('kana-systems-logo.png'))
    await user.type(screen.getByLabelText(/workspace password/i), 'test-password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    await screen.findByRole('heading', { name: /your contract workspace/i })
    expect(screen.getByText('Legal', { selector: 'b' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Kana Systems' })).toBeInTheDocument()
    expect(screen.queryByText('Acquisition Lens')).not.toBeInTheDocument()
  })
  it('authenticates and loads only persisted documents without seeded samples', async () => {
    const fetch = mockApi([doc])
    await login()
    expect(await screen.findByText(doc.title)).toBeInTheDocument()
    const call = fetch.mock.calls.find(([url]) => String(url).endsWith('/documents'))
    expect((call?.[1]?.headers as Record<string, string>).Authorization).toBe('Bearer test-token')
    expect(fetch.mock.calls.some(([url]) => String(url).includes('/demo/'))).toBe(false)
    expect(localStorage).toHaveLength(0)
    expect(sessionStorage).toHaveLength(0)
  })
  it('validates input and saves the actual supplied text before reviewing', async () => {
    const fetch = mockApi([], (url, init) => url.endsWith('/documents') && init?.method === 'POST' ? json(doc) : undefined)
    const user = await login()
    await user.click(screen.getByRole('button', { name: /add contract/i }))
    await user.type(screen.getByLabelText(/document title/i), doc.title)
    await user.type(screen.getByLabelText(/^contract text$/i), 'Too short')
    await user.click(screen.getByRole('button', { name: /save to workspace/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/at least 80 characters/i)
    await user.clear(screen.getByLabelText(/^contract text$/i))
    await user.type(screen.getByLabelText(/^contract text$/i), doc.text)
    await user.click(screen.getByRole('button', { name: /save to workspace/i }))
    expect(await screen.findByRole('heading', { name: doc.title })).toBeInTheDocument()
    const save = fetch.mock.calls.find(([url, init]) => String(url).endsWith('/documents') && init?.method === 'POST')
    expect(JSON.parse(save?.[1]?.body as string)).toEqual({ title: doc.title, text: doc.text, category: 'contract' })
    expect(screen.getByRole('button', { name: /set up review/i })).toBeInTheDocument()
  })
  it('fails closed when authentication is unavailable', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'))
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByLabelText(/workspace password/i), 'password')
    await user.click(screen.getByRole('button', { name: /enter workspace/i }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/local api is unavailable/i)
    expect(screen.queryByRole('navigation')).not.toBeInTheDocument()
  })
  it('navigates the six connected workspaces and distinguishes catalog from indexed data', async () => {
    mockApi()
    const user = await login()
    await user.click(screen.getByRole('button', { name: /source library/i }))
    await screen.findByRole('heading', { name: /source library/i })
    await user.click(screen.getByRole('button', { name: /available sources/i }))
    expect(await screen.findByText('FAR Part 52')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /data connections/i }))
    expect(await screen.findByText(/dynamodb/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /entities & relationships/i }))
    expect(await screen.findByRole('heading', { name: /entities & relationships/i })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /reviewed records/i }))
    expect(await screen.findByRole('heading', { name: /reviewed records/i })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /target objects/i })).not.toBeInTheDocument()
  })
  it('renders saved findings safely and retains contract context in follow-up questions', async () => {
    const hostile = '<img src=x onerror="alert(1)">'
    const questions: unknown[] = []
    const fetch = mockApi([doc], (url, init) => {
      if (url.endsWith('/reviews')) return json([{ id: 'review-1', document_id: doc.id, document_version: doc.version, decision: 'draft', revision: 1, note: '', analysis: { ...analysis, findings: [{ ...analysis.findings[0], title: hostile }] } }])
      if (url.endsWith('/questions')) {
        if (init?.method === 'POST') questions.push({ id: 'question-1', document_id: doc.id, query: 'What warranty applies?', response: { answer: 'Grounded answer from the service.', synthesis_mode: 'test-engine', statements: [], evidence: [] } })
        return json(questions)
      }
    })
    const user = await login()
    await user.click(await screen.findByText(doc.title))
    expect(await screen.findByRole('heading', { name: hostile })).toBeInTheDocument()
    expect(document.querySelector('img[src="x"]')).toBeNull()
    await user.click(screen.getByRole('tab', { name: /ask this contract/i }))
    await user.type(screen.getByLabelText(/question or drafting request/i), 'What warranty applies?')
    await user.click(screen.getByRole('button', { name: /ask \/ generate/i }))
    expect(await screen.findByText('Grounded answer from the service.')).toBeInTheDocument()
    const call = fetch.mock.calls.find(([url, init]) => String(url).endsWith('/questions') && init?.method === 'POST')
    expect(JSON.parse(call?.[1]?.body as string)).toMatchObject({ document_id: doc.id, query: 'What warranty applies?', mode: 'answer' })
  })
  it('clears credentials on sign out', async () => {
    const fetch = mockApi()
    const user = await login()
    await user.click(screen.getByRole('button', { name: /sign out/i }))
    await waitFor(() => expect(screen.getByLabelText(/workspace password/i)).toBeInTheDocument())
    await apiClient.getSample()
    const call = fetch.mock.calls.at(-1)
    expect((call?.[1]?.headers as Record<string, string>).Authorization).toBeUndefined()
  })
})
