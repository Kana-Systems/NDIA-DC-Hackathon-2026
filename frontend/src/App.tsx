import { useEffect, useId, useRef, useState } from 'react'
import {
  ArrowRight, BookOpen, CheckCircle2, ChevronDown, CircleAlert, FileCheck2,
  FileText, Info, KeyRound, Library, Link2, LoaderCircle, LockKeyhole, Network,
  RotateCcw, SearchCheck, ShieldCheck, Sparkles, TriangleAlert,
} from 'lucide-react'
import { apiClient, demoModeEnabled } from './api/client'
import type { AcquisitionMetadata, AnalysisResponse, DemoDocument, Finding, RiskLevel, SourceRecord } from './types'
import { MetadataForm } from './MetadataForm'
import { emptyMetadata } from './types'
import { EvidenceGraph } from './EvidenceGraph'
import {
  FoundationsWorkspace, IngestionWorkspace, IntelligenceQueryWorkspace,
} from './IntelligenceWorkspaces'

type View = 'review' | 'query' | 'ingestion' | 'foundations' | 'sources'

const riskLabel: Record<RiskLevel, string> = {
  critical: 'Critical', high: 'High', medium: 'Medium', low: 'Low', info: 'Informational',
}

function Brand() {
  return (
    <div className="brand" aria-label="Acquisition Lens">
      <span className="brand-mark" aria-hidden="true"><SearchCheck size={22} /></span>
      <span>Acquisition <b>Lens</b></span>
    </div>
  )
}

function Login({ onLogin }: { onLogin: (password: string) => Promise<void> }) {
  const [password, setPassword] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const errorId = useId()

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!password) return
    setPending(true)
    setError('')
    try { await onLogin(password) }
    catch (err) { setError(err instanceof Error ? err.message : 'Unable to sign in. Please try again.') }
    finally { setPending(false) }
  }

  return (
    <main className="login-shell">
      <section className="login-story" aria-labelledby="login-title">
        <Brand />
        <div className="eyebrow"><Sparkles size={15} /> Source-grounded contract intelligence</div>
        <h1 id="login-title">See the obligation<br />before you sign it.</h1>
        <p className="login-lede">Review federal contract language against acquisition rules and surface what deserves a closer look—in minutes, not days.</p>
        <div className="value-list" aria-label="Product benefits">
          <div><span><FileCheck2 /></span><p><strong>Traceable findings</strong>Every flag connects to source material.</p></div>
          <div><span><Network /></span><p><strong>Connected context</strong>See relationships across clauses, rules, and risks.</p></div>
          <div><span><ShieldCheck /></span><p><strong>Human judgment stays central</strong>Confidence and uncertainty are always visible.</p></div>
        </div>
      </section>
      <section className="login-panel" aria-label="Secure access">
        <div className="login-card">
          <span className="lock-icon" aria-hidden="true"><LockKeyhole size={25} /></span>
          <p className="kicker">Secure workspace</p>
          <h2>Enter your access password</h2>
          <p>Use public or synthetic documents for this hackathon workspace.</p>
          <form onSubmit={submit}>
            <label htmlFor="password">Workspace password</label>
            <div className="password-wrap"><KeyRound size={18} aria-hidden="true" /><input id="password" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} aria-describedby={error ? errorId : undefined} autoFocus /></div>
            {error && <p className="form-error" id={errorId} role="alert"><CircleAlert size={16} />{error}</p>}
            <button className="primary-button login-button" disabled={!password || pending}>
              {pending ? <><LoaderCircle className="spin" size={18} />Verifying…</> : <>Enter workspace <ArrowRight size={18} /></>}
            </button>
          </form>
          {demoModeEnabled && <p className="demo-hint"><span>Explicit demo mode enabled</span> Use the password configured in <code>VITE_DEMO_PASSWORD</code> when the API is offline.</p>}
          <div className="secure-note"><ShieldCheck size={16} /> Demo workspace · Processing follows the backend configuration</div>
        </div>
      </section>
    </main>
  )
}

function Header({ active, onNavigate }: { active: View; onNavigate: (view: View) => void }) {
  const links: { view: View; label: string }[] = [
    { view: 'review', label: 'Contract review' },
    { view: 'query', label: 'Cited intelligence' },
    { view: 'ingestion', label: 'Ingestion' },
    { view: 'foundations', label: 'Foundations' },
    { view: 'sources', label: 'Source library' },
  ]
  return (
    <header className="app-header">
      <Brand />
      <nav aria-label="Primary navigation">
        {links.map(link => <button key={link.view} aria-current={active === link.view ? 'page' : undefined} className={active === link.view ? 'active' : ''} onClick={() => onNavigate(link.view)}>{link.label}</button>)}
      </nav>
      <div className="header-status"><span aria-hidden="true" /> Demo workspace</div>
    </header>
  )
}

function RiskBadge({ risk }: { risk: RiskLevel }) {
  return <span className={`risk-badge risk-${risk}`}><span aria-hidden="true" />{riskLabel[risk]} risk</span>
}

function FindingCard({ finding, index }: { finding: Finding; index: number }) {
  return (
    <article className="finding-card">
      <div className="finding-top">
        <span className="finding-number">{String(index + 1).padStart(2, '0')}</span>
        <div>
          <div className="finding-meta"><span>{finding.category}</span><RiskBadge risk={finding.risk} /><span className="confidence">{finding.confidence == null ? "Classifier score unavailable" : `${Math.round(finding.confidence * 100)}% classifier score`}</span></div>
          <h3>{finding.title}</h3>
        </div>
      </div>
      <p className="mini-label">Grounding: {finding.groundingStatus ?? "unverified"}</p><blockquote>{finding.excerpt || "No matching document passage; review the missing-clause rationale."}</blockquote>
      <div className="finding-grid">
        <div><p className="mini-label">Why it matters</p><p>{finding.explanation}</p></div>
        <div className="recommendation"><p className="mini-label"><CheckCircle2 size={15} /> Recommended next step</p><p>{finding.recommendation}</p></div>
      </div>
      <details className="source-details">
        <summary><Link2 size={15} /> {finding.sources.length} source{finding.sources.length === 1 ? '' : 's'} used <ChevronDown size={16} className="chevron" /></summary>
        {finding.sources.map((source, sourceIndex) => (
          <a href={source.url || undefined} target="_blank" rel="noreferrer" key={`${finding.id}-${sourceIndex}-${source.url}`}>
            <BookOpen size={16} />
            <span><strong>{source.title}</strong><small>{source.citation ?? 'Open source'}</small><small>{source.excerpt}</small></span>
            <span className={source.verificationStatus === 'retrieved_evidence' ? 'source-state verified' : 'source-state candidate'}>{source.verificationStatus === 'retrieved_evidence' ? 'Retrieved evidence' : 'Verify citation'}</span>
          </a>
        ))}
      </details>
    </article>
  )
}


function Results({ analysis, onReset }: { analysis: AnalysisResponse; onReset: () => void }) {
  const titleRef = useRef<HTMLHeadingElement>(null)
  useEffect(() => { titleRef.current?.focus() }, [])
  const counts = analysis.findings.reduce<Record<string, number>>((all, finding) => ({ ...all, [finding.risk]: (all[finding.risk] ?? 0) + 1 }), {})
  const manifest = analysis.report?.corpus_manifest
  const corpusSummary = manifest
    && typeof manifest.documents === 'number'
    && typeof manifest.chunks === 'number'
    && typeof manifest.links === 'number'
    ? { documents: manifest.documents, chunks: manifest.chunks, links: manifest.links, retrievedAt: manifest.retrieved_at }
    : null
  return (
    <>
      <div className="results-heading">
        <div><p className="kicker">Review complete</p><h1 ref={titleRef} tabIndex={-1}>Analysis results</h1><p>{analysis.analyzedAt ?? 'Analysis generated from the current source catalog.'}</p></div>
        <button className="secondary-button" onClick={onReset}><RotateCcw size={16} /> New review</button>
      </div>
      <section className="summary-card" aria-labelledby="summary-title">
        <div className="risk-score"><div className={`score-ring ${analysis.overallRisk}`}><strong>{analysis.findings.length}</strong><span>review items</span></div><div><p className="mini-label">Overall assessment</p><h2 id="summary-title">{riskLabel[analysis.overallRisk]} review priority</h2><RiskBadge risk={analysis.overallRisk} /></div></div>
        <p className="summary-copy">{analysis.documentSummary}</p>
        {analysis.disclaimer && <p className="analysis-disclaimer">{analysis.disclaimer}</p>}
        <div className="summary-stats"><div><strong>{analysis.findings.length}</strong><span>Findings</span></div><div><strong>{counts.critical ?? 0}</strong><span>Critical</span></div><div><strong>{counts.high ?? 0}</strong><span>High</span></div><div><strong>{counts.medium ?? 0}</strong><span>Medium</span></div></div>
      </section>
      {analysis.report && <section className="report-details" aria-label="Review provenance">
        <h2>Review provenance and clause inventory</h2>
        <p>Explanation mode: <strong>{analysis.report.synthesis_mode}</strong></p>
        <p>Classifiers: {analysis.report.classifier_model_ids.join(', ') || 'Not reported'}. Heuristic and keyword identifiers indicate rule-based processing.</p>
        <p>No overall legal-confidence score is calculated. Clause detection and applicability require separate review.</p>
        {corpusSummary && <p>Indexed corpus: {corpusSummary.documents.toLocaleString()} source documents, {corpusSummary.chunks.toLocaleString()} passages, and {corpusSummary.links.toLocaleString()} source references.{corpusSummary.retrievedAt ? ` Ingested ${corpusSummary.retrievedAt}.` : ''}</p>}
        <details><summary>Inspect {analysis.report.clause_status_inventory.length} clause statuses</summary>
          {analysis.report.clause_status_inventory.map((clause, index) => <article key={`${clause.clause_id}-${index}`}><h3>{clause.clause_id} — {clause.title}</h3><p><strong>{clause.status}</strong>: {clause.rationale}</p></article>)}
        </details>
      </section>}
      <div className="results-layout">
        <section className="findings-list" aria-labelledby="findings-title">
          <div className="list-title"><div><p className="kicker">Prioritized review queue</p><h2 id="findings-title">Source-backed findings</h2></div><span>{analysis.findings.length} items</span></div>
          {analysis.findings.map((finding, index) => <FindingCard finding={finding} index={index} key={finding.id} />)}
        </section>
        <aside><EvidenceGraph analysis={analysis} /></aside>
      </div>
    </>
  )
}

function ReviewWorkspace({ onAnalyze, onLoadSample }: { onAnalyze: (title: string, text: string, metadata: AcquisitionMetadata, file?: File) => Promise<void>; onLoadSample: () => Promise<DemoDocument> }) {
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [metadata, setMetadata] = useState<AcquisitionMetadata>({ ...emptyMetadata })
  const [file, setFile] = useState<File | undefined>()
  const uploadRef = useRef<HTMLInputElement>(null)
  const [pending, setPending] = useState(false)
  const [samplePending, setSamplePending] = useState(false)
  const [error, setError] = useState('')
  const characterCount = text.trim().length

  async function loadSample() {
    setSamplePending(true); setError('')
    try {
      const sample = await onLoadSample()
      setTitle(sample.title); setText(sample.text)
      setFile(undefined)
      if (uploadRef.current) uploadRef.current.value = ''
      if (sample.metadata) setMetadata(sample.metadata)
    } catch (err) { setError(err instanceof Error ? err.message : 'The judge-ready sample is unavailable.') }
    finally { setSamplePending(false) }
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (!file && characterCount < 80) { setError('Add at least 80 characters of contract language to run a meaningful review.'); return }
    if (![metadata.agency, metadata.solicitation_number, metadata.place_of_performance].every(v => v.trim())) {
      setError('Enter agency, solicitation number, and place of performance, or load the sample.'); return
    }
    setPending(true); setError('')
    try { await onAnalyze(title || 'Untitled contract review', text, metadata, file) }
    catch (err) { setError(err instanceof Error ? err.message : 'Analysis failed. Check the service and try again.') }
    finally { setPending(false) }
  }

  return (
    <>
      <section className="workspace-hero">
        <div><div className="eyebrow"><Sparkles size={15} /> Source-grounded analysis</div><h1>Find the clause that changes the deal.</h1><p>Paste federal contract language. Acquisition Lens maps obligations to authoritative sources, prioritizes risk, and gives your team a clear review path.</p></div>
        <div className="trust-stat"><strong>4</strong><span>review dimensions</span><small>risk · authority · confidence · action</small></div>
      </section>
      <section className="mission-coverage" aria-label="Supported use cases">
        <article><span>01</span><h2>Legal contract review</h2><p>Legal-BERT identifies clause candidates while governed rules and FAR/DFARS evidence drive missing-language and risk recommendations.</p></article>
        <article><span>02</span><h2>Governed RAG intelligence</h2><p>Natural-language answers, summaries, and analyst drafts retain source citations, grounding status, and access-control boundaries.</p></article>
        <article><span>03</span><h2>Foundational intelligence</h2><p>Versioned ingestion, entity resolution, change detection, provenance, and analyst quality control support reference-product maintenance.</p></article>
      </section>
      <section className="review-card" aria-labelledby="review-title">
        <div className="review-card-header"><div><p className="kicker">New analysis</p><h2 id="review-title">Review contract language</h2></div><button className="sample-button" onClick={() => void loadSample()} disabled={samplePending}>{samplePending ? <><LoaderCircle className="spin" size={15} /> Loading sample…</> : <><Sparkles size={15} /> Load judge-ready sample</>}</button></div>
        <form onSubmit={submit}>
          <MetadataForm value={metadata} onChange={setMetadata} />
          <label htmlFor="document-upload">Upload PDF or DOCX (optional, up to 15 MB)</label>
          <input id="document-upload" ref={uploadRef} type="file" accept=".pdf,.docx" onChange={event => {
            const chosen = event.target.files?.[0]
            if (chosen && chosen.size > 15 * 1024 * 1024) { setError('Choose a file smaller than 15 MB.'); event.target.value = ''; setFile(undefined); return }
            setFile(chosen); setError('')
            if (chosen) setText('')
          }} />
          {file && <p>{file.name} selected. <button type="button" className="sample-button" onClick={() => { setFile(undefined); if (uploadRef.current) uploadRef.current.value = '' }}>Use pasted text instead</button></p>}
          <label htmlFor="document-title">Document title <span>Optional</span></label>
          <input id="document-title" className="text-input" value={title} onChange={(event) => setTitle(event.target.value)} placeholder="e.g., Technology modernization RFP — Section I" />
          <div className="textarea-label"><label htmlFor="contract-text">Contract or solicitation text</label><span className={characterCount > 0 && characterCount < 80 ? 'short' : ''}>{characterCount.toLocaleString()} characters</span></div>
          <div className="textarea-wrap">
            <textarea id="contract-text" disabled={Boolean(file)} value={text} onChange={(event) => setText(event.target.value)} placeholder={'Paste clauses, a statement of work, or solicitation language here…\n\nFor the strongest demo, load the sample document.'} aria-describedby={error ? 'analysis-error' : 'privacy-note'} />
            {!text && <div className="drop-cue" aria-hidden="true"><FileText size={18} /> Plain text analysis</div>}
          </div>
          {error && <p className="form-error" id="analysis-error" role="alert"><CircleAlert size={16} />{error}</p>}
          <div className="review-actions"><p id="privacy-note"><LockKeyhole size={15} /> Public or synthetic documents only</p><button className="primary-button analyze-button" disabled={pending}>{pending ? <><LoaderCircle className="spin" size={18} />Analyzing clauses…</> : <>Analyze contract <ArrowRight size={18} /></>}</button></div>
        </form>
      </section>
      <div className="how-it-works" aria-label="Analysis workflow">
        <div><span>01</span><p><strong>Identify</strong>Clause families and obligations</p></div><ArrowRight />
        <div><span>02</span><p><strong>Connect</strong>Candidate rules and sources</p></div><ArrowRight />
        <div><span>03</span><p><strong>Prioritize</strong>Risks and human review actions</p></div>
      </div>
    </>
  )
}

function SourceLibrary({ sources, loading, error, onRetry }: { sources: SourceRecord[]; loading: boolean; error: string; onRetry: () => void }) {
  return (
    <section className="source-page" aria-labelledby="library-title">
      <div className="source-page-heading"><div><p className="kicker">Evidence catalog</p><h1 id="library-title">Authoritative sources</h1><p>Source directory for the requested federal regulations, award data, decisions, and research datasets. Listing a source does not mean it has been ingested or used in a finding.</p></div><span className="catalog-count"><strong>{sources.length}</strong> cataloged sources</span></div>
      {loading && <div className="state-card" role="status"><LoaderCircle className="spin" /><h2>Loading source catalog</h2><p>Connecting to the acquisition knowledge layer…</p></div>}
      {error && <div className="state-card error-state" role="alert"><CircleAlert /><h2>Couldn’t load sources</h2><p>{error}</p><button className="secondary-button" onClick={onRetry}>Try again</button></div>}
      {!loading && !error && <div className="source-grid">{sources.map((source) => <a className="source-card" href={source.url} target="_blank" rel="noreferrer" key={source.id}><span className="source-icon"><Library /></span><div><span className="source-type">{source.type}</span><h2>{source.title}</h2><p>{source.organization}</p><span className={`authority-tag ${source.status.toLowerCase()}`}><CheckCircle2 />{source.status}</span></div><ArrowRight className="source-arrow" /></a>)}</div>}
      <div className="catalog-note"><Info /><div><strong>Citation integrity</strong><p>A source link establishes provenance, not applicability. Exact clause, prescription, agency supplement, deviation, and contract context require human confirmation.</p></div></div>
    </section>
  )
}

export default function App() {
  const [authenticated, setAuthenticated] = useState(false)
  const [demoMode, setDemoMode] = useState(false)
  const [view, setView] = useState<View>('review')
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null)
  const [sources, setSources] = useState<SourceRecord[]>([])
  const [sourcesLoading, setSourcesLoading] = useState(false)
  const [sourcesError, setSourcesError] = useState('')

  async function login(password: string) {
    const result = await apiClient.login(password)
    setDemoMode(result.demoMode)
    setAuthenticated(result.data.authenticated)
  }
  async function analyze(title: string, text: string, metadata: AcquisitionMetadata, file?: File) {
    const result = file ? await apiClient.uploadFile(file, metadata) : await apiClient.analyze(title, text, metadata)
    setDemoMode((current) => current || result.demoMode)
    setAnalysis(result.data)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }
  async function loadSample() {
    const result = await apiClient.getSample()
    setDemoMode((current) => current || result.demoMode)
    return result.data
  }
  async function loadSources() {
    setSourcesLoading(true); setSourcesError('')
    try {
      const result = await apiClient.getSources()
      setSources(result.data); setDemoMode((current) => current || result.demoMode)
    } catch (error) { setSourcesError(error instanceof Error ? error.message : 'The source catalog is unavailable.') }
    finally { setSourcesLoading(false) }
  }
  if (!authenticated) return <Login onLogin={login} />
  return (
    <div className="app-shell">
      <Header active={view} onNavigate={(nextView) => {
        setView(nextView)
        if (nextView === 'review') setAnalysis(null)
        if (nextView === 'sources' && !sources.length) void loadSources()
      }} />
      {demoMode && <div className="demo-banner" role="status"><span><Info size={15} /> Local demo mode</span><p>The backend is offline, so deterministic sample data is powering this experience.</p></div>}
      <main className="main-content">
        {view === 'review' && (analysis ? <Results analysis={analysis} onReset={() => setAnalysis(null)} /> : <ReviewWorkspace onAnalyze={analyze} onLoadSample={loadSample} />)}
        {view === 'query' && <IntelligenceQueryWorkspace offline={demoMode} />}
        {view === 'ingestion' && <IngestionWorkspace offline={demoMode} />}
        {view === 'foundations' && <FoundationsWorkspace offline={demoMode} />}
        {view === 'sources' && <SourceLibrary sources={sources} loading={sourcesLoading} error={sourcesError} onRetry={loadSources} />}
      </main>
      <footer className="legal-footer"><div><TriangleAlert size={18} /><p><strong>Human review required.</strong> Acquisition Lens supports issue spotting and research. It is not legal advice and does not determine clause applicability, compliance, or contract acceptability.</p></div><span>Acquisition Lens · Local hackathon demonstration</span></footer>
    </div>
  )
}
