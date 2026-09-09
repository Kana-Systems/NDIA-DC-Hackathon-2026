import { useState } from "react";
import { MessageSquare, Search } from "lucide-react";
import type { Review } from "./workspaceApi";
import { Empty } from "./WorkspaceShared";

type Finding = Review["analysis"]["findings"][number];
const priority = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
const linked = (finding: Finding) => finding.grounding_status === "verified" && finding.citations.length > 0;

export function FindingList({ findings, onAsk }: {
  findings: Finding[];
  onAsk?: (id: string) => void;
}) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const urgent = findings.filter(f => priority[f.severity] < 2).length;
  const unverified = findings.filter(f => !linked(f)).length;
  const visible = findings.filter(f =>
    (filter === "all" || (filter === "urgent" ? priority[f.severity] < 2 : !linked(f))) &&
    `${f.title} ${f.category} ${f.excerpt} ${f.explanation} ${f.recommendation}`.toLowerCase().includes(query.trim().toLowerCase())
  ).sort((a, b) => priority[a.severity] - priority[b.severity]);

  return <section aria-label="Finding triage" className="finding-triage">
    <div className="triage-toolbar panel">
      <div className="filter-buttons" role="group" aria-label="Filter findings">
        {[["all", "All findings", findings.length], ["urgent", "High priority", urgent], ["unverified", "Needs evidence", unverified]].map(([key, label, count]) =>
          <button key={key} className="quiet" aria-pressed={filter === key} onClick={() => setFilter(String(key))}>
            {label} <span className="count">{count}</span>
          </button>
        )}
      </div>
      <label className="search-box"><Search size={16} aria-hidden="true" />
        <input aria-label="Search findings" placeholder="Search a clause, issue, or recommendation" value={query} onChange={e => setQuery(e.target.value)} />
      </label>
      <p className="muted" role="status">Showing {visible.length} of {findings.length} findings · Highest priority first</p>
    </div>
    {!visible.length && <Empty title={findings.length ? "No findings match these filters" : "No findings returned"}>
      {findings.length ? "Choose All findings or clear your search to see the other items." : "Check the clause inventory and source coverage before recording your decision. This is not a compliance determination."}
    </Empty>}
    {visible.map(f => <article key={f.id} className="panel finding">
      <div className="section-heading">
        <span className={`severity ${f.severity}`}>{f.severity}</span>
        <span className={`evidence-label ${linked(f) ? "linked" : "unverified"}`}>{linked(f) ? "Citation linked" : "Evidence needs review"}</span>
      </div>
      <small>{f.category}</small>
      <h2>{f.title}</h2>
      <blockquote>{f.excerpt || "No matching contract passage was returned. Check the original document."}</blockquote>
      <h3>Why it matters</h3><p>{f.explanation}</p>
      <h3>Recommended next step</h3><p>{f.recommendation}</p>
      <details>
        <summary>{f.citations.length} supporting sources</summary>
        {!f.citations.length && <p className="notice">No supporting source was linked. Verify the issue against an authoritative source before relying on it.</p>}
        {f.citations.map((c, index) => <blockquote key={index}>
          <strong>{c.title} · {c.section}</strong><p>{c.excerpt}</p>
          {c.url && /^https?:\/\//i.test(c.url) && <a href={c.url} target="_blank" rel="noreferrer">Open original source</a>}
        </blockquote>)}
      </details>
      {onAsk && <button className="quiet" onClick={() => onAsk(f.id)}><MessageSquare size={16} />Ask about this finding</button>}
    </article>)}
  </section>;
}
