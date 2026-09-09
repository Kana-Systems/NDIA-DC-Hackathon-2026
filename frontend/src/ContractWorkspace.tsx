import { useCallback, useEffect, useRef, useState } from "react";
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
import { Chip, Decision, Empty, ErrorNotice, WorkspaceTabs } from "./WorkspaceShared";
import { FindingList } from "./FindingList";
import { message } from "./workspaceUtils";

export function Research({
  documentId,
  findingId,
  documents,
  onClearFinding,
}: {
  documentId: string | null;
  findingId?: string;
  documents: Document[];
  onClearFinding?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("answer");
  const [history, setHistory] = useState<Question[]>([]);
  const [busy, setBusy] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const queryRef = useRef<HTMLTextAreaElement>(null);
  const [savedQuestions, setSavedQuestions] = useState<string[]>([]);
  const reload = useCallback(
    () =>
      workspace
        .questions()
        .then((rows) =>
          setHistory(rows.filter((row) => row.document_id === documentId && (!findingId || row.finding_id === findingId))),
        )
        .catch((e) => setError(message(e))),
    [documentId, findingId],
  );
  useEffect(() => {
    let active = true;
    void workspace
      .questions()
      .then((rows) => {
        if (active)
          setHistory(rows.filter((row) => row.document_id === documentId && (!findingId || row.finding_id === findingId)));
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, [documentId, findingId]);
  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (busy) return;
    setBusy(true);
    setGenerating(true);
    setError("");
    setNotice("");
    try {
      await workspace.ask(query, mode, documentId, findingId);
      setQuery("");
      await reload();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
      setGenerating(false);
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
        {findingId && onClearFinding && <button className="text-button" onClick={onClearFinding}>All contract questions</button>}
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
        <div className="research-starters" role="group" aria-label="Research starting points">
          {[
            ["Summarize obligations", "summary", "Summarize the obligations, responsible parties, and deadlines in the available sources. Cite each statement and identify what is missing."],
            ["Find evidence gaps", "answer", "Which material claims or requirements need more supporting evidence? Identify gaps and cite the available sources without assuming missing facts."],
            ["Draft a decision memo", "draft", "Draft a decision memo with the mission question, key findings, supporting citations, uncertainties, and recommended next actions for a human reviewer."],
          ].map(([label, output, prompt]) => <button key={label} type="button" className="quiet" disabled={busy} onClick={() => {
            setMode(output); setQuery(prompt); queryRef.current?.focus();
          }}>{label}</button>)}
        </div>
        <label>
          Question or drafting request
          <textarea
            ref={queryRef}
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
        {generating && <p role="status" className="generation-status">Retrieving sources and preparing a cited draft. Your question will be saved with the result.</p>}
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
          <div className="answer-provenance">
            <span>{q.response.evidence.length} sources</span>
            <span>{q.response.statements.filter(s => s.grounding_status === "verified").length} of {q.response.statements.length} statements citation-linked</span>
            {q.created_at && <time dateTime={q.created_at}>{new Date(q.created_at).toLocaleString()}</time>}
          </div>
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
            disabled={busy || !q.response.statements.length || savedQuestions.includes(q.id)}
            onClick={() => {
              setBusy(true);
              void workspace
                .record(q.id, q.query.slice(0, 200))
                .then(() => {
                  setSavedQuestions(ids => [...ids, q.id]);
                  setNotice(
                    "Draft saved to Reviewed records. Analyst approval is required before export.",
                  );
                })
                .catch((e) => setError(message(e)))
                .finally(() => setBusy(false));
            }}
          >
            <Layers size={16} />
            {savedQuestions.includes(q.id) ? "Saved to Reviewed records" : "Save as reviewable record"}
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
  const [operation, setOperation] = useState("");
  const busy = Boolean(operation);
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
    if (busy) return;
    setOperation("save");
    setError("");
    try {
      const updated = await workspace.metadata(doc.id, metadata, doc.revision);
      setDoc(updated);
      if (run) {
        setOperation("review");
        await workspace.review(doc.id);
        setTab("findings");
      }
      await reload();
      onChanged();
    } catch (e) {
      setError(message(e));
    } finally {
      setOperation("");
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
            disabled={busy || doc.available === false || doc.status !== "ready"}
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
      {doc.available === false && doc.status === "ready" && <p className="notice">
        Source access needs a fresh sync. Open Sources → Data connections and sync the connection before reviewing.
      </p>}
      {busy && <div className="notice work-progress" role="status">
        <LoaderCircle className="spin" size={18} />
        <div><strong>{operation === "review" ? "Reviewing contract" : operation === "index" ? "Retrying source indexing" : "Saving acquisition details"}</strong>
          {operation === "review" && <p>Checking clauses, retrieving evidence, and preparing findings. This can take a few minutes.</p>}
        </div>
      </div>}
      {(doc.status === "index-failed" || doc.status === "indexing") && <section className="notice">
        <p>This document is saved, but search indexing is incomplete.</p>
        <button disabled={busy} onClick={() => {
          setOperation("index"); void workspace.reindex(doc.id).then(() => reload())
            .catch(e => setError(message(e))).finally(() => setOperation(""));
        }}>Retry indexing</button>
      </section>}
      <WorkspaceTabs label="Contract workspace" value={tab} onChange={setTab} tabs={[
        ["findings", "Findings"], ["document", "Document"],
        ["research", "Ask this contract"], ["details", "Acquisition details"],
        ["history", "Review history"],
      ]}>
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
              disabled={busy || doc.available === false || doc.status !== "ready"}
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
          onClearFinding={() => setFindingId(undefined)}
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
              <FindingList key={latest.id} findings={latest.analysis.findings} onAsk={id => {
                setFindingId(id); setTab("research");
              }} />
            </section>
            <aside>
              <section className="panel">
                <Decision key={`${latest.id}:${latest.revision}`} item={latest} onChange={() => void reload()} />
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
      </WorkspaceTabs>
    </>
  );
}
