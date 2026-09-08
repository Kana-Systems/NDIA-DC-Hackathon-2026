import { useEffect, useMemo, useState } from 'react'
import {
  ArrowRight, CheckCircle2, CircleAlert, Database, Download, FileOutput,
  GitMerge, LoaderCircle, RefreshCw, Search, ShieldCheck, XCircle,
} from 'lucide-react'
import { apiClient } from './api/client'
import type {
  ChangeEvent, EntityAttribute, GenerationMode, IngestionStatus,
  IntelligenceEntity, IntelligenceResponse, Relationship, ReviewDecision,
  TargetObject, TargetObjectExport,
} from './types'

function ErrorMessage({ message }: Readonly<{ message: string }>) {
  return message ? <p className="form-error workflow-error" role="alert"><CircleAlert size={16} />{message}</p> : null
}

function statusClass(status: ReviewDecision | 'verified' | 'unverified') {
  return `status-chip status-${status}`
}

export function IntelligenceQueryWorkspace({ offline }: Readonly<{ offline: boolean }>) {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<GenerationMode>('answer')
  const [workflow, setWorkflow] = useState('mission-support')
  const [sourceTypes, setSourceTypes] = useState('')
  const [documentIds, setDocumentIds] = useState('')
  const [entityIds, setEntityIds] = useState('')
  const [effectiveAfter, setEffectiveAfter] = useState('')
  const [result, setResult] = useState<IntelligenceResponse | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')

  const split = (value: string) => value.split(',').map(item => item.trim()).filter(Boolean)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setPending(true); setError(''); setResult(null)
    try {
      setResult(await apiClient.queryIntelligence({
        query, mode, workflow,
        filters: {
          source_types: split(sourceTypes),
          document_ids: split(documentIds),
          entity_ids: split(entityIds),
          effective_after: effectiveAfter || null,
        },
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The intelligence query failed.')
    } finally { setPending(false) }
  }

  return (
    <section className="workflow-page" aria-labelledby="query-title">
      <div className="workflow-hero">
        <div><p className="kicker">Cited intelligence</p><h1 id="query-title">Query and draft from governed evidence.</h1><p>Generate an answer, summary, or draft. Returned statements retain citation IDs and visible grounding status.</p></div>
        <span className="workflow-icon"><Search /></span>
      </div>
      {offline && <OfflineNotice />}
      <div className="workflow-layout">
        <form className="workflow-card workflow-form" onSubmit={submit}>
          <label htmlFor="intel-query">Intelligence request</label>
          <textarea id="intel-query" required minLength={2} maxLength={4000} value={query} onChange={event => setQuery(event.target.value)} placeholder="Ask a source-grounded question or describe the product to draft…" />
          <div className="form-grid">
            <label>Generation mode<select value={mode} onChange={event => setMode(event.target.value as GenerationMode)}><option value="answer">Answer</option><option value="summary">Summary</option><option value="draft">Draft</option></select></label>
            <label>Workflow<input value={workflow} required maxLength={100} onChange={event => setWorkflow(event.target.value)} /></label>
            <label>Source types <span>Comma-separated</span><input value={sourceTypes} onChange={event => setSourceTypes(event.target.value)} /></label>
            <label>Document IDs <span>Comma-separated</span><input value={documentIds} onChange={event => setDocumentIds(event.target.value)} /></label>
            <label>Entity IDs <span>Comma-separated</span><input value={entityIds} onChange={event => setEntityIds(event.target.value)} /></label>
            <label>Effective after<input type="date" value={effectiveAfter} onChange={event => setEffectiveAfter(event.target.value)} /></label>
          </div>
          <ErrorMessage message={error} />
          <button type="submit" className="primary-button" disabled={pending || query.trim().length < 2}>{pending ? <><LoaderCircle className="spin" size={17} />Generating…</> : <>Run cited query <ArrowRight size={17} /></>}</button>
        </form>
        <section className="workflow-card result-panel" aria-live="polite" aria-label="Intelligence response">
          {!result && <EmptyState icon={<FileOutput />} title="No generated product" text="Submit a live query to see cited results. Offline mode intentionally supplies no J2 output." />}
          {result && <>
            <div className="result-meta"><span>{result.mode}</span><span>{result.workflow}</span><span>{result.synthesis_mode}</span></div>
            <h2>Generated product</h2>
            <p className="answer-copy">{result.answer}</p>
            <h3>Cited statements</h3>
            <div className="statement-list">{result.statements.map((statement, index) => <article key={`${result.response_id}-${index}`}><span className={statusClass(statement.grounding_status)}>{statement.grounding_status}</span><p>{statement.text}</p><small>Citations: {statement.citation_ids.join(', ') || 'None returned'}</small></article>)}</div>
            <h3>Evidence</h3>
            <div className="evidence-list">{result.evidence.map(item => <article key={item.evidence_id}><strong>{item.title}</strong><span>{item.source} · {item.security_label}</span><p>{item.excerpt}</p><small>{item.evidence_id} · document {item.document_id || 'not reported'}{item.version ? ` · version ${item.version}` : ''}</small>{item.url && <a href={item.url} target="_blank" rel="noreferrer">Open source</a>}</article>)}</div>
          </>}
        </section>
      </div>
    </section>
  )
}

function OfflineNotice() {
  return <div className="offline-notice" role="note"><ShieldCheck /><div><strong>Live API required</strong><p>Offline demo mode is active. These screens remain available for review, but no intelligence, entity, status, or target-object results are fabricated.</p></div></div>
}

function EmptyState({ icon, title, text }: Readonly<{ icon: React.ReactNode; title: string; text: string }>) {
  return <div className="workflow-empty">{icon}<h2>{title}</h2><p>{text}</p></div>
}

export function IngestionWorkspace({ offline }: Readonly<{ offline: boolean }>) {
  const [status, setStatus] = useState<IngestionStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function load() {
    setLoading(true); setError('')
    try { setStatus(await apiClient.getIngestionStatus()) }
    catch (err) { setError(err instanceof Error ? err.message : 'Ingestion status is unavailable.') }
    finally { setLoading(false) }
  }
  useEffect(() => {
    if (offline) return
    let active = true
    void apiClient.getIngestionStatus()
      .then(next => { if (active) setStatus(next) })
      .catch(err => { if (active) setError(err instanceof Error ? err.message : 'Ingestion status is unavailable.') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [offline])

  return (
    <section className="workflow-page" aria-labelledby="ingestion-title">
      <div className="workflow-hero"><div><p className="kicker">Ingestion operations</p><h1 id="ingestion-title">Evidence pipeline status.</h1><p>Inspect configuration readiness and durable document counts reported by the live service.</p></div><span className="workflow-icon"><Database /></span></div>
      {offline && <OfflineNotice />}
      <div className="toolbar"><button className="secondary-button" onClick={() => void load()} disabled={loading}>{loading ? <LoaderCircle className="spin" size={16} /> : <RefreshCw size={16} />} Refresh status</button></div>
      <ErrorMessage message={error} />
      {loading && !status && <output className="state-card"><LoaderCircle className="spin" /><h2>Checking ingestion services</h2></output>}
      {!loading && !status && !error && <EmptyState icon={<Database />} title="No status loaded" text="Connect to the live API and refresh to inspect ingestion readiness." />}
      {status && <div className="metric-grid" aria-live="polite">
        <article><span>Security domain</span><strong>{status.security_domain}</strong></article>
        <article><span>Fixture documents expected</span><strong>{status.fixture_documents_expected.toLocaleString()}</strong></article>
        <article><span>Durable documents</span><strong>{status.durable_documents == null ? 'Unavailable' : status.durable_documents.toLocaleString()}</strong></article>
        <article><span>Durable store</span><strong>{status.durable_store_configured ? 'Configured' : 'Not configured'}</strong></article>
        <article><span>Graph connector</span><strong>{status.graph_connector_configured ? 'Configured' : 'Not configured'}</strong></article>
      </div>}
    </section>
  )
}

export function FoundationsWorkspace({ offline }: Readonly<{ offline: boolean }>) {
  const [entities, setEntities] = useState<IntelligenceEntity[]>([])
  const [changes, setChanges] = useState<ChangeEvent[]>([])
  const [relationships, setRelationships] = useState<Relationship[]>([])
  const [name, setName] = useState('')
  const [entityType, setEntityType] = useState('')
  const [aliases, setAliases] = useState('')
  const [relationshipSpecs, setRelationshipSpecs] = useState('')
  const [attributeName, setAttributeName] = useState('')
  const [attributeValue, setAttributeValue] = useState('')
  const [citationId, setCitationId] = useState('')
  const [documentId, setDocumentId] = useState('')
  const [note, setNote] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')

  async function load() {
    setPending(true); setError('')
    try {
      const [nextEntities, nextChanges, nextRelationships] = await Promise.all([
        apiClient.getEntities(), apiClient.getChanges(), apiClient.getRelationships(),
      ])
      setEntities(nextEntities); setChanges(nextChanges); setRelationships(nextRelationships)
    } catch (err) { setError(err instanceof Error ? err.message : 'Foundational intelligence is unavailable.') }
    finally { setPending(false) }
  }
  useEffect(() => {
    if (offline) return
    let active = true
    void Promise.all([apiClient.getEntities(), apiClient.getChanges(), apiClient.getRelationships()])
      .then(([nextEntities, nextChanges, nextRelationships]) => {
        if (!active) return
        setEntities(nextEntities); setChanges(nextChanges); setRelationships(nextRelationships)
      })
      .catch(err => { if (active) setError(err instanceof Error ? err.message : 'Foundational intelligence is unavailable.') })
    return () => { active = false }
  }, [offline])

  async function resolve(event: React.FormEvent) {
    event.preventDefault(); setPending(true); setError('')
    const attributes: Record<string, EntityAttribute> = {}
    if (attributeName.trim()) attributes[attributeName.trim()] = attributeValue
    try {
      await apiClient.resolveEntities([{
        name, entity_type: entityType, attributes,
        aliases: aliases.split(',').map(item => item.trim()).filter(Boolean),
        relationships: relationshipSpecs.split(',').map(item => item.trim()).filter(Boolean),
        provenance: citationId.trim() ? [{ citation_id: citationId.trim(), document_id: documentId.trim(), version: '', content_sha256: '' }] : [],
      }])
      setName(''); setEntityType(''); setAliases(''); setRelationshipSpecs(''); setAttributeName(''); setAttributeValue(''); setCitationId(''); setDocumentId('')
      await load()
    } catch (err) { setError(err instanceof Error ? err.message : 'Entity resolution failed.'); setPending(false) }
  }

  async function decide(entityId: string, decision: Exclude<ReviewDecision, 'draft'>) {
    setPending(true); setError('')
    try {
      const updated = await apiClient.decideEntity(entityId, decision, note)
      setEntities(current => current.map(item => item.entity_id === updated.entity_id ? updated : item))
    } catch (err) { setError(err instanceof Error ? err.message : 'The entity decision failed.') }
    finally { setPending(false) }
  }

  const entityNames = useMemo(() => new Map(entities.map(item => [item.entity_id, item.canonical_name])), [entities])

  return (
    <section className="workflow-page" aria-labelledby="foundations-title">
      <div className="workflow-hero"><div><p className="kicker">Foundational intelligence</p><h1 id="foundations-title">Resolve entities and review change.</h1><p>Manage canonical entities, relationship provenance, detected changes, and analyst decisions.</p></div><span className="workflow-icon"><GitMerge /></span></div>
      {offline && <OfflineNotice />}
      <div className="workflow-layout foundations-layout">
        <form className="workflow-card workflow-form" onSubmit={resolve}>
          <div className="card-heading"><div><p className="kicker">Entity resolution</p><h2>Submit a candidate</h2></div><button type="button" className="icon-button" aria-label="Refresh foundational intelligence" onClick={() => void load()} disabled={pending}><RefreshCw size={16} /></button></div>
          <label>Name<input required maxLength={300} value={name} onChange={event => setName(event.target.value)} /></label>
          <label>Entity type<input required maxLength={100} value={entityType} onChange={event => setEntityType(event.target.value)} placeholder="organization, facility, system…" /></label>
          <label>Aliases <span>Comma-separated</span><input value={aliases} onChange={event => setAliases(event.target.value)} /></label>
          <label>Relationships <span>relation:target, comma-separated</span><input value={relationshipSpecs} onChange={event => setRelationshipSpecs(event.target.value)} /></label>
          <div className="form-grid"><label>Attribute name<input value={attributeName} onChange={event => setAttributeName(event.target.value)} /></label><label>Attribute value<input value={attributeValue} onChange={event => setAttributeValue(event.target.value)} /></label><label>Citation ID<input value={citationId} onChange={event => setCitationId(event.target.value)} /></label><label>Document ID<input value={documentId} onChange={event => setDocumentId(event.target.value)} /></label></div>
          <button type="submit" className="primary-button" disabled={pending}>{pending ? <LoaderCircle className="spin" size={16} /> : <GitMerge size={16} />} Resolve candidate</button>
        </form>
        <section className="workflow-card">
          <div className="card-heading"><div><p className="kicker">Analyst queue</p><h2>Canonical entities</h2></div><span>{entities.length}</span></div>
          <label htmlFor="entity-note">Decision note</label><textarea id="entity-note" className="compact-textarea" value={note} onChange={event => setNote(event.target.value)} maxLength={2000} />
          <ErrorMessage message={error} />
          {!entities.length && !pending && <EmptyState icon={<GitMerge />} title="No entities returned" text="Resolve a candidate or refresh the live repository." />}
          <div className="record-list">{entities.map(entity => <article key={entity.entity_id}>
            <div><span className={statusClass(entity.review_status)}>{entity.review_status}</span><small>{entity.entity_type}</small></div>
            <h3>{entity.canonical_name}</h3>
            <p>{Object.entries(entity.attributes).map(([key, value]) => `${key}: ${String(value)}`).join(' · ') || 'No attributes'}</p>
            <small>Aliases: {entity.aliases.join(', ') || 'None'} · Provenance: {entity.provenance.length}</small>
            <div className="inline-actions"><button className="approve-button" disabled={pending} onClick={() => void decide(entity.entity_id, 'approved')}><CheckCircle2 size={15} />Approve</button><button className="reject-button" disabled={pending} onClick={() => void decide(entity.entity_id, 'rejected')}><XCircle size={15} />Reject</button></div>
          </article>)}</div>
        </section>
      </div>
      <div className="data-sections">
        <section className="workflow-card"><div className="card-heading"><h2>Detected changes</h2><span>{changes.length}</span></div>{changes.length ? <div className="record-list">{changes.map(change => <article key={change.change_id}><h3>{entityNames.get(change.entity_id) ?? change.entity_id}</h3><p>Changed: {change.changed_fields.join(', ')}</p><small>{new Date(change.detected_at).toLocaleString()} · {change.provenance.length} provenance links</small></article>)}</div> : <p className="muted-copy">No detected changes returned.</p>}</section>
        <section className="workflow-card"><div className="card-heading"><h2>Relationships</h2><span>{relationships.length}</span></div>{relationships.length ? <div className="record-list">{relationships.map(item => <article key={item.relationship_id}><span className={statusClass(item.grounding_status)}>{item.grounding_status}</span><h3>{entityNames.get(item.source_entity_id) ?? item.source_entity_id} → {entityNames.get(item.target_entity_id) ?? item.target_entity_id}</h3><p>{item.relationship_type}</p><small>Citations: {item.citation_ids.join(', ') || 'None'}</small></article>)}</div> : <p className="muted-copy">No relationships returned.</p>}</section>
      </div>
    </section>
  )
}

export function TargetObjectsWorkspace({ offline }: Readonly<{ offline: boolean }>) {
  const [entities, setEntities] = useState<IntelligenceEntity[]>([])
  const [entityId, setEntityId] = useState('')
  const [objectType, setObjectType] = useState('target-system-object')
  const [fields, setFields] = useState<string[]>(['name'])
  const [objects, setObjects] = useState<TargetObject[]>([])
  const [note, setNote] = useState('')
  const [exported, setExported] = useState<TargetObjectExport | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const allowedFields = ['name', 'country', 'status', 'category']

  async function loadEntities() {
    setPending(true); setError('')
    try {
      const next = await apiClient.getEntities()
      setEntities(next); setEntityId(current => current || next[0]?.entity_id || '')
    } catch (err) { setError(err instanceof Error ? err.message : 'Entities are unavailable.') }
    finally { setPending(false) }
  }
  useEffect(() => {
    if (offline) return
    let active = true
    void apiClient.getEntities()
      .then(next => {
        if (!active) return
        setEntities(next)
        setEntityId(next[0]?.entity_id || '')
      })
      .catch(err => { if (active) setError(err instanceof Error ? err.message : 'Entities are unavailable.') })
    return () => { active = false }
  }, [offline])

  async function draft(event: React.FormEvent) {
    event.preventDefault(); setPending(true); setError(''); setExported(null)
    try {
      const created = await apiClient.createTargetObject({ object_type: objectType, entity_id: entityId, requested_fields: fields })
      setObjects(current => [created, ...current])
    } catch (err) { setError(err instanceof Error ? err.message : 'Target object drafting failed.') }
    finally { setPending(false) }
  }
  async function decide(objectId: string, decision: Exclude<ReviewDecision, 'draft'>) {
    setPending(true); setError(''); setExported(null)
    try {
      const updated = await apiClient.decideTargetObject(objectId, decision, note)
      setObjects(current => current.map(item => item.object_id === objectId ? updated : item))
    } catch (err) { setError(err instanceof Error ? err.message : 'The analyst decision failed.') }
    finally { setPending(false) }
  }
  async function exportObject(objectId: string) {
    setPending(true); setError('')
    try { setExported(await apiClient.exportTargetObject(objectId)) }
    catch (err) { setError(err instanceof Error ? err.message : 'The target object could not be exported.') }
    finally { setPending(false) }
  }

  return (
    <section className="workflow-page" aria-labelledby="targets-title">
      <div className="workflow-hero"><div><p className="kicker">Target objects</p><h1 id="targets-title">Draft, decide, then export.</h1><p>Transform a resolved entity into a schema-bound object. Export remains blocked until analyst approval succeeds.</p></div><span className="workflow-icon"><FileOutput /></span></div>
      {offline && <OfflineNotice />}
      <div className="workflow-layout">
        <form className="workflow-card workflow-form" onSubmit={draft}>
          <div className="card-heading"><div><p className="kicker">Draft builder</p><h2>New target object</h2></div><button type="button" className="icon-button" aria-label="Refresh entities" onClick={() => void loadEntities()}><RefreshCw size={16} /></button></div>
          <label>Object type<input required value={objectType} onChange={event => setObjectType(event.target.value)} /></label>
          <label>Resolved entity<select required value={entityId} onChange={event => setEntityId(event.target.value)}><option value="">Select an entity</option>{entities.map(entity => <option key={entity.entity_id} value={entity.entity_id}>{entity.canonical_name} — {entity.entity_type}</option>)}</select></label>
          <fieldset className="field-picker"><legend>Requested fields</legend>{allowedFields.map(field => <label key={field}><input type="checkbox" checked={fields.includes(field)} onChange={event => setFields(current => event.target.checked ? [...current, field] : current.filter(item => item !== field))} />{field}</label>)}</fieldset>
          <button type="submit" className="primary-button" disabled={pending || !entityId}><FileOutput size={16} />Draft object</button>
        </form>
        <section className="workflow-card">
          <div className="card-heading"><div><p className="kicker">Analyst gate</p><h2>Draft queue</h2></div><span>{objects.length}</span></div>
          <label htmlFor="target-note">Decision note</label><textarea id="target-note" className="compact-textarea" value={note} onChange={event => setNote(event.target.value)} maxLength={2000} />
          <ErrorMessage message={error} />
          {!objects.length && <EmptyState icon={<FileOutput />} title="No drafted objects" text="Choose a resolved entity and request fields to create a live draft." />}
          <div className="record-list">{objects.map(object => <article key={object.object_id}>
            <div><span className={statusClass(object.status)}>{object.status}</span><small>{object.object_type}</small></div>
            <h3>{entities.find(entity => entity.entity_id === object.entity_id)?.canonical_name ?? object.entity_id}</h3>
            <div className="target-fields">{object.fields.map(field => <span key={field.name}><strong>{field.name}</strong>{String(field.value ?? 'Not supplied')} <em>{field.grounding_status}</em></span>)}</div>
            <div className="inline-actions"><button className="approve-button" disabled={pending || object.status !== 'draft'} onClick={() => void decide(object.object_id, 'approved')}><CheckCircle2 size={15} />Approve</button><button className="reject-button" disabled={pending || object.status !== 'draft'} onClick={() => void decide(object.object_id, 'rejected')}><XCircle size={15} />Reject</button><button className="secondary-button" disabled={pending || object.status !== 'approved'} onClick={() => void exportObject(object.object_id)}><Download size={15} />Export JSON</button></div>
          </article>)}</div>
        </section>
      </div>
      {exported && <section className="workflow-card export-panel" aria-live="polite"><div className="card-heading"><div><p className="kicker">Safe export</p><h2>Export payload</h2></div><span>{exported.adapter}</span></div><p><ShieldCheck size={15} /> External write performed: <strong>{exported.external_write_performed ? 'Yes' : 'No'}</strong></p><pre>{JSON.stringify(exported, null, 2)}</pre></section>}
    </section>
  )
}
