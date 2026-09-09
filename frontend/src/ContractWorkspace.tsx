import { useCallback, useEffect, useState } from "react";
import {
  ArrowLeft,
  ChevronRight,
  Layers,
  Link2,
  LoaderCircle,
  MessageSquare,
  Search,
} from "lucide-react";
import { MetadataForm } from "./MetadataForm";
import { emptyMetadata } from "./types";
import type { AcquisitionMetadata } from "./types";
import { workspace } from "./workspaceApi";
import type { Document, Review, Question } from "./workspaceApi";
import { Chip, Decision, Empty, ErrorNotice } from "./WorkspaceShared";
import { message } from "./workspaceUtils";

export function Research({
  documentId,
  findingId,
  documents,
}: {
  documentId: string | null;
  findingId?: string;
  documents: Document[];
}) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("answer");
  const [history, setHistory] = useState<Question[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const reload = useCallback(
    () =>
      workspace
        .questions()
        .then((rows) =>
          setHistory(rows.filter((row) => row.document_id === documentId)),
        )
        .catch((e) => setError(message(e))),
    [documentId],
  );
  useEffect(() => {
    let active = true;
    void workspace
      .questions()
      .then((rows) => {
        if (active)
          setHistory(rows.filter((row) => row.document_id === documentId));
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, [documentId]);
  async function ask(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await workspace.ask(query, mode, documentId, findingId);
      setQuery("");
      await reload();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="research">
      <div className="context-strip">
        <Link2 size={16} />
        <span>
          {documentId
            ? `Context: ${documents.find((d) => d.id === documentId)?.title || "Selected contract"}`
            : "Context: your indexed sources and official regulatory corpus"}
        </span>
        {findingId && <Chip>Selected finding</Chip>}
      </div>
      <form className="panel" onSubmit={ask}>
        <div className="section-heading">
          <h2>Ask the evidence</h2>
          <label className="inline-label">
            Output
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="answer">Answer</option>
              <option value="summary">Summary</option>
              <option value="draft">Review memo draft</option>
            </select>
          </label>
        </div>
        <label>
          Question or drafting request
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            minLength={2}
            maxLength={3500}
            required
            placeholder="What does this provision require, and which sources support that interpretation?"
          />
        </label>
        <p className="muted">
          Generated text is a draft. Citation checks confirm references, not
          legal correctness.
        </p>
        <ErrorNotice error={error} />
        <button disabled={busy || query.trim().length < 2}>
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <MessageSquare size={16} />
          )}
          Ask / generate
        </button>
      </form>
      {notice && (
        <p role="status" className="notice">
          {notice}
        </p>
      )}
      {!history.length && (
        <Empty title="A question starts the trail">
          Answers and drafts are saved with the evidence used to produce them.
        </Empty>
      )}
      {history.map((q) => (
        <article className="panel answer-card" key={q.id}>
          <div className="section-heading">
            <h3>{q.query}</h3>
            <Chip>{q.response.synthesis_mode}</Chip>
          </div>
          <p className="answer-text">{q.response.answer}</p>
          <details>
            <summary>Inspect statements and citations</summary>
            {q.response.statements.map((s, i) => (
              <div className="statement" key={i}>
                <Chip>
                  {s.grounding_status === "verified"
                    ? "Citation linked"
                    : "Unverified"}
                </Chip>
                <p>{s.text}</p>
                <small>{s.citation_ids.join(", ")}</small>
              </div>
            ))}
            {q.response.evidence.map((e) => (
              <blockquote key={e.evidence_id}>
                <strong>{e.title}</strong>
                <p>{e.excerpt}</p>
                <small>
                  {e.evidence_id} · {e.version}
                </small>
                {e.url && /^https?:\/\//.test(e.url) && (
                  <a href={e.url} target="_blank" rel="noreferrer">
                    Open source
                  </a>
                )}
              </blockquote>
            ))}
          </details>
          <button
            className="quiet"
            disabled={busy || !q.response.statements.length}
            onClick={() => {
              setBusy(true);
              void workspace
                .record(q.id, q.query.slice(0, 200))
                .then(() =>
                  setNotice(
                    "Draft saved to Reviewed records. Analyst approval is required before export.",
                  ),
                )
                .catch((e) => setError(message(e)))
                .finally(() => setBusy(false));
            }}
          >
            <Layers size={16} />
            Save as reviewable record
          </button>
        </article>
      ))}
    </section>
  );
}

export function ContractWorkspace({
  initial,
  documents,
  onBack,
  onChanged,
}: {
  initial: Document;
  documents: Document[];
  onBack: () => void;
  onChanged: () => void;
}) {
  const [doc, setDoc] = useState(initial);
  const [metadata, setMetadata] = useState<AcquisitionMetadata>(
    initial.metadata || emptyMetadata,
  );
  const [reviews, setReviews] = useState<Review[]>([]);
  const [tab, setTab] = useState("findings");
  const [findingId, setFindingId] = useState<string>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const reload = useCallback(async () => {
    try {
      const [updated, all] = await Promise.all([
        workspace.document(initial.id),
        workspace.reviews(),
      ]);
      setDoc(updated);
      setReviews(all.filter((r) => r.document_id === initial.id));
    } catch (e) {
      setError(message(e));
    }
  }, [initial.id]);
  useEffect(() => {
    let active = true;
    void Promise.all([workspace.document(initial.id), workspace.reviews()])
      .then(([updated, all]) => {
        if (active) {
          setDoc(updated);
          setMetadata(updated.metadata || emptyMetadata);
          setReviews(all.filter((r) => r.document_id === initial.id));
        }
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, [initial.id]);
  const latest = reviews[0];
  async function saveAndReview(run: boolean) {
    setBusy(true);
    setError("");
    try {
      const updated = await workspace.metadata(doc.id, metadata, doc.revision);
      setDoc(updated);
      if (run) {
        await workspace.review(doc.id);
        setTab("findings");
      }
      await reload();
      onChanged();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <button className="back-link" onClick={onBack}>
        <ArrowLeft size={15} />
        All contracts
      </button>
      <div className="page-heading">
        <div>
          <p className="overline">Contract workspace</p>
          <h1>{doc.title}</h1>
          <p>
            Version {doc.version} · Saved{" "}
            {new Date(doc.updated_at).toLocaleString()}
          </p>
        </div>
        {doc.category === "contract" && (
          <button
            disabled={busy}
            onClick={() => {
              if (!doc.metadata) setTab("details");
              else void saveAndReview(true);
            }}
          >
            {busy ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Search size={17} />
            )}
            {doc.metadata ? "Run review" : "Set up review"}
          </button>
        )}
      </div>
      <div className="step-strip">
        <span className="done">01 Document saved</span>
        <ChevronRight />
        <span className={latest ? "done" : ""}>02 Evidence & findings</span>
        <ChevronRight />
        <span className={latest?.decision === "approved" ? "done" : ""}>
          03 Human decision
        </span>
      </div>
      <ErrorNotice error={error} />
      {(doc.status === "index-failed" || doc.status === "indexing") && <section className="notice">
        <p>This document is saved, but search indexing is incomplete.</p>
        <button disabled={busy} onClick={() => {
          setBusy(true); void workspace.reindex(doc.id).then(() => reload())
            .catch(e => setError(message(e))).finally(() => setBusy(false));
        }}>Retry indexing</button>
      </section>}
      <div className="tabs" role="tablist" aria-label="Contract workspace">
        {[
          ["findings", "Findings"],
          ["document", "Document"],
          ["research", "Ask this contract"],
          ["details", "Acquisition details"],
          ["history", "Review history"],
        ].map(([key, label]) => (
          <button
            role="tab"
            aria-selected={tab === key}
            className={tab === key ? "active" : ""}
            key={key}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "details" && (
        <form
          className="panel"
          onSubmit={(e) => {
            e.preventDefault();
            void saveAndReview(false);
          }}
        >
          <h2>Acquisition details</h2>
          <p>
            These details guide applicability screening. They require your
            confirmation.
          </p>
          <MetadataForm value={metadata} onChange={setMetadata} />
          <div className="actions">
            <button disabled={busy}>Save details</button>
            <button
              type="button"
              className="quiet"
              disabled={busy}
              onClick={() => void saveAndReview(true)}
            >
              Save and run review
            </button>
          </div>
        </form>
      )}
      {tab === "document" && (
        <section className="panel">
          <div className="section-heading">
            <h2>Extracted document</h2>
            <Chip>{doc.category}</Chip>
          </div>
          <p className="muted">
            Original source:{" "}
            {doc.source_key || "Uploaded / pasted by workspace owner"} · Access:{" "}
            {doc.owner}
          </p>
          <pre className="document-text">{doc.text}</pre>
        </section>
      )}
      {tab === "research" && (
        <Research
          key={`${doc.id}:${findingId || ""}`}
          documentId={doc.id}
          findingId={findingId}
          documents={documents}
        />
      )}
      {tab === "history" && (
        <section className="panel">
          <h2>Saved review history</h2>
          {reviews.length ? (
            reviews.map((r) => (
              <article className="history-row" key={r.id}>
                <div>
                  <strong>{new Date(r.created_at).toLocaleString()}</strong>
                  <p>
                    Document {r.document_version} · {r.analysis.engine}
                  </p>
                  <small>{r.note}</small>
                  <details>
                    <summary>Inspect saved findings</summary>
                    {r.analysis.findings.map((f) => (
                      <p key={f.id}>
                        <strong>{f.title}</strong>
                        <br />
                        {f.explanation}
                      </p>
                    ))}
                  </details>
                </div>
                <Chip>{r.decision}</Chip>
              </article>
            ))
          ) : (
            <p>No reviews yet.</p>
          )}
        </section>
      )}
      {tab === "findings" &&
        (!latest ? (
          <Empty title="Your document is ready">
            Confirm acquisition details, then run a review. Findings are
            generated from your document—not seeded examples.
          </Empty>
        ) : (
          <div className="review-layout">
            <section>
              <div className="panel review-summary">
                <div className="section-heading">
                  <h2>{latest.analysis.findings.length} items for review</h2>
                  <Chip>{latest.analysis.overall_risk} priority</Chip>
                </div>
                <p>{latest.analysis.document_summary}</p>
                <small>
                  Engine: {latest.analysis.engine} ·{" "}
                  {latest.analysis.report?.classifier_model_ids?.join(", ")}
                </small>
                {latest.document_version !== doc.version && (
                  <p className="notice error">
                    The source has changed. Run a new review before export.
                  </p>
                )}
              </div>
              {latest.analysis.findings.map((f) => (
                <article key={f.id} className="panel finding">
                  <div className="section-heading">
                    <span className={`severity ${f.severity}`}>
                      {f.severity}
                    </span>
                    <small>{f.category}</small>
                  </div>
                  <h2>{f.title}</h2>
                  <blockquote>
                    {f.excerpt || "No matching passage was returned."}
                  </blockquote>
                  <h3>Why it matters</h3>
                  <p>{f.explanation}</p>
                  <h3>Recommended next step</h3>
                  <p>{f.recommendation}</p>
                  <details>
                    <summary>{f.citations.length} supporting sources</summary>
                    {f.citations.map((c, i) => (
                      <blockquote key={i}>
                        <strong>
                          {c.title} · {c.section}
                        </strong>
                        <p>{c.excerpt}</p>
                        {c.url && /^https?:\/\//.test(c.url) && (
                          <a href={c.url} target="_blank" rel="noreferrer">
                            Open original source
                          </a>
                        )}
                      </blockquote>
                    ))}
                  </details>
                  <button
                    className="quiet"
                    onClick={() => {
                      setFindingId(f.id);
                      setTab("research");
                    }}
                  >
                    <MessageSquare size={16} />
                    Ask about this finding
                  </button>
                </article>
              ))}
            </section>
            <aside>
              <section className="panel">
                <Decision item={latest} onChange={() => void reload()} />
              </section>
              <section className="panel">
                <h3>Evidence, not certainty</h3>
                <p>{latest.analysis.disclaimer}</p>
                <details>
                  <summary>Clause inventory</summary>
                  {latest.analysis.report?.clause_status_inventory?.map(
                    (c, i) => (
                      <p key={i}>
                        <strong>
                          {c.clause_id} · {c.status}
                        </strong>
                        <br />
                        {c.rationale}
                      </p>
                    ),
                  )}
                </details>
              </section>
            </aside>
          </div>
        ))}
    </>
  );
}
