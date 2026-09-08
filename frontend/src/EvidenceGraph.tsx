import { useState } from 'react'
import { BookOpen, FileText, Network } from 'lucide-react'
import type { AnalysisResponse } from './types'

export function EvidenceGraph({ analysis }: { analysis: AnalysisResponse }) {
  const [selected, setSelected] = useState<string | null>(null)
  const finding = analysis.findings.find(item => item.id === selected)
  const links = analysis.report?.knowledge_graph?.source_references ?? []
  return <section className="graph-card" aria-labelledby="graph-title">
    <h2 id="graph-title"><Network size={20} /> Evidence relationships</h2>
    <p className="graph-instruction">Select a finding to see the passages it cites. Each connection comes from this review.</p>
    <div className="evidence-root"><FileText size={18} /> Reviewed document</div>
    <div className="evidence-branches">{analysis.findings.map(item =>
      <button key={item.id} className={`evidence-branch ${selected === item.id ? 'selected' : ''}`} aria-pressed={selected === item.id} onClick={() => setSelected(item.id)}>
        {item.title}<small>{item.sources.length} cited passages · {item.groundingStatus ?? 'unverified'}</small>
      </button>,
    )}</div>
    {finding && <div className="evidence-passages"><h3>{finding.title}</h3>
      {finding.sources.length === 0 && <p>No retrieved citation supports this finding.</p>}
      {finding.sources.map((source, index) => <div key={`${source.url}-${index}`}>
        <BookOpen size={15} /> <a href={source.url || undefined} target="_blank" rel="noreferrer">{source.title}</a>
        <p>{source.excerpt || source.citation}</p>
      </div>)}
    </div>}
    {links.length > 0 && <details><summary>{links.length} source-document references</summary>
      <ul>{links.map((link, index) => <li key={index}>{link.source} → {link.target}</li>)}</ul>
    </details>}
  </section>
}
