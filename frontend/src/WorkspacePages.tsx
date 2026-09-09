import { useCallback, useEffect, useState } from "react";
import { Database, Download, ExternalLink, Plus, RefreshCw, Search } from "lucide-react";
import { workspace } from "./workspaceApi";
import type {
  AuditEvent,
  Connections,
  Document,
  Entity,
  Library,
  Review,
  StructuredRecord,
} from "./workspaceApi";
import { Chip, CompactList, Decision, Empty, ErrorNotice } from "./WorkspaceShared";
import { downloadJson, message, sourceStatus } from "./workspaceUtils";
import { FindingList } from "./FindingList";

export function ConnectionsPage({ onChanged }: { onChanged: () => void }) {
  const [data, setData] = useState<Connections>();
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [name, setName] = useState("");
  const [folder, setFolder] = useState("");
  const [providerChoice, setProvider] = useState("");
  const provider = providerChoice || (data?.sharepoint_available ? "sharepoint" : "shared-folder");
  const [category, setCategory] = useState("reference");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sharepointStatus, setSharepointStatus] = useState("");
  const load = useCallback(async () => {
    try {
      const [result, history] = await Promise.all([
        workspace.connections(),
        workspace.events(),
      ]);
      setData(result);
      setEvents(history);
    } catch (e) {
      setError(message(e));
    }
  }, []);
  useEffect(() => {
    let active = true;
    void Promise.all([workspace.connections(), workspace.events()])
      .then(([result, history]) => {
        if (active) {
          setData(result);
          setEvents(history);
        }
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, []);
  const syncing = data?.connections.some((c) => c.status === "syncing");
  useEffect(() => {
    if (!syncing) return;
    const timer = window.setInterval(() => {
      void load();
      onChanged();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [syncing, load, onChanged]);
  async function connect(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await workspace.connect({ name, provider, folder, category });
      setName("");
      await load();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  async function sync(id: string) {
    setBusy(true);
    setError("");
    try {
      await workspace.sync(id);
      await load();
      onChanged();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="overline">A traceable starting point.</p>
          <h1>Data connections</h1>
          <p>
            Connect approved sources. Inspect what changed, what failed, and
            what is ready to use.
          </p>
        </div>
        <button className="quiet" onClick={() => void load()}>
          <RefreshCw size={16} />
          Refresh
        </button>
      </div>
      <ErrorNotice error={error} />
      <div className="status-bar">
        <Chip>{data?.persistence || "Checking persistence"}</Chip>
        <Chip>{data?.security_domain || "Checking domain"}</Chip>
        {data?.document_storage && <Chip>Documents: {data.document_storage}</Chip>}
        {data?.search && <Chip>Search: {data.search}</Chip>}
        <span>
          {data?.automatic_sync_seconds
            ? `Automatic sync: ${data.automatic_sync_seconds}s`
            : "Manual sync · scheduled worker not enabled"}
        </span>
      </div>
      <div className="connections-layout">
        <div className="connection-setup">
        <section className="panel sharepoint-check">
          <h2>SharePoint connection check</h2>
          <p>Credentials stay in AWS Secrets Manager. This checks the selected site and library read access before any files are imported.</p>
          <button disabled={busy || !data?.sharepoint_available} onClick={() => {
            setBusy(true); setError(""); setSharepointStatus("");
            void workspace.checkSharePoint().then(r => setSharepointStatus(`${r.site} / ${r.library}: connected`))
              .catch(e => setError(message(e))).finally(() => setBusy(false));
          }}>Test SharePoint access</button>
          {sharepointStatus && <p role="status">{sharepointStatus}</p>}
          {!data?.sharepoint_available && <p>Administrator setup required: app registration, selected-site read grant, and AWS connector secret.</p>}
        </section>
        <form className="panel" onSubmit={connect}>
          <h2>Add a connection</h2>
          <label>
            Name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              maxLength={150}
            />
          </label>
          <label>
            Source type
            <select
              value={provider}
              onChange={(e) => { setProvider(e.target.value); setFolder(""); }}
            >
              <option value="shared-folder">Approved shared folder</option>
              <option value="sharepoint">Configured SharePoint drive</option>
            </select>
          </label>
          {provider === "shared-folder" ? (
            <>
              <label>
                Folder within approved import root
                <input
                  value={folder}
                  onChange={(e) => setFolder(e.target.value)}
                  placeholder="e.g. procurement/contracts"
                />
              </label>
              <p className="muted">
                {data?.shared_folder_available
                  ? "Import root configured. Only its subfolders are accessible."
                  : "An administrator must set WORKSPACE_IMPORT_ROOT before a folder can be connected."}
              </p>
            </>
          ) : (
            <p className="muted">
              {data?.sharepoint_available
                ? "Uses the configured drive and Microsoft Graph permissions. No credentials are entered here."
                : "Not configured. An administrator must supply approved Microsoft Graph access. No credentials are stored in the browser."}
            </p>
          )}
          <label>
            Document purpose
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {[
                "contract",
                "reference",
                "policy",
                "playbook",
                "clause-library",
              ].map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          <button
            disabled={
              busy ||
              (provider === "shared-folder"
                ? !data?.shared_folder_available
                : !data?.sharepoint_available)
            }
          >
            <Plus size={16} />
            Save connection
          </button>
        </form>
        </div>
        <section className="connected-sources">
          <h2>Connected sources</h2>
          {!data?.connections.length && (
            <Empty title="No connections yet">
              Upload individual documents in Contracts or configure an approved
              source here.
            </Empty>
          )}
          <CompactList label="connections">{data?.connections.map((c) => (
            <article className="panel" key={c.id}>
              <div className="section-heading">
                <h3>{c.name}</h3>
                <Chip>{c.status}</Chip>
              </div>
              <p>
                {c.provider} · {c.category}
              </p>
              <p className="muted">
                Last sync:{" "}
                {c.last_sync ? new Date(c.last_sync).toLocaleString() : "Never"}
              </p>
              <div className="sync-counts">
                {Object.entries(c.counts).map(([key, value]) => (
                  <span key={key}>
                    <strong>{value}</strong>
                    {key}
                  </span>
                ))}
              </div>
              <button
                disabled={busy || c.status === "syncing"}
                onClick={() => void sync(c.id)}
              >
                <RefreshCw size={16} />
                {c.status === "syncing" ? "Syncing…" : "Sync now"}
              </button>
              {c.errors.map((e, i) => (
                <p className="notice error" key={i}>
                  {e.file}: {e.message}
                </p>
              ))}
            </article>
          ))}</CompactList>
        </section>
      </div>
      <section className="panel">
        <h2>Ingestion & change history</h2>
        <CompactList label="events">{events
          .filter(
            (e) =>
              e.action.startsWith("document.") ||
              e.action.startsWith("connection."),
          )
          .map((e) => (
            <div className="history-row" key={e.id}>
              <strong>{e.action.replaceAll(".", " ")}</strong>
              <small>{e.record_id.slice(0, 10)}</small>
              <time>{new Date(e.created_at).toLocaleString()}</time>
            </div>
          ))}</CompactList>
      </section>
    </>
  );
}

export function LibraryPage({ onOpen }: { onOpen: (doc: Document) => void }) {
  const [data, setData] = useState<Library>();
  const [tab, setTab] = useState("indexed");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("policy");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(async (q = "") => {
    try {
      setData(await workspace.library(q));
    } catch (e) {
      setError(message(e));
    }
  }, []);
  useEffect(() => {
    let active = true;
    void workspace
      .library()
      .then((result) => {
        if (active) setData(result);
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, []);
  async function add(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await workspace.create({ title, text, category, source_url: url });
      setTitle("");
      setText("");
      setUrl("");
      setTab("indexed");
      await load();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="overline">Know what your answer rests on.</p>
          <h1>Source library</h1>
          <p>
            Inspect source readiness and provenance before using a document as
            evidence. Browse the catalog to discover additional references.
          </p>
        </div>
      </div>
      <ErrorNotice error={error} />
      <div className="tabs">
        {[
          ["indexed", "Indexed documents"],
          ["catalog", "Available sources"],
          ["add", "Add reference"],
        ].map(([key, label]) => (
          <button
            className={tab === key ? "active" : ""}
            key={key}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "indexed" && (
        <>
          <section className="panel">
            <div className="section-heading">
              <h2>Official FAR / DFARS snapshot</h2>
              <Chip>{data?.corpus ? "Indexed" : "Unavailable"}</Chip>
            </div>
            {data?.corpus && (
              <p>
                {data.corpus.documents.toLocaleString()} source documents ·{" "}
                {data.corpus.chunks.toLocaleString()} passages · Ingested{" "}
                {new Date(data.corpus.retrieved_at).toLocaleDateString()}
              </p>
            )}
            <form
              className="search-form"
              onSubmit={(e) => {
                e.preventDefault();
                void load(query);
              }}
            >
              <input
                aria-label="Search official passages"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Clause number or topic"
              />
              <button>
                <Search size={16} />
                Search
              </button>
            </form>
            <CompactList key={query} label="passages">{data?.evidence.map((e) => (
              <blockquote key={e.evidence_id}>
                <strong>{e.title}</strong>
                <p>{e.excerpt}</p>
                {e.url && /^https?:\/\//.test(e.url) && (
                  <a href={e.url} target="_blank" rel="noreferrer">
                    Open original source
                  </a>
                )}
                <small>Version {e.version}</small>
              </blockquote>
            ))}</CompactList>
          </section>
          <section className="panel">
            <h2>Your indexed documents</h2>
            {!data?.documents.length && (
              <p>
                No workspace documents yet. The official corpus above is
                separate.
              </p>
            )}
            <CompactList label="documents">{data?.documents.map((d) => (
              <button
                className="contract-row"
                key={d.id}
                disabled={d.status === "source-error"}
                onClick={() => onOpen(d)}
              >
                <Database size={18} />
                <span>
                  <strong>{d.title}</strong>
                  <small>
                    {d.category} · Version {d.version} · Owner {d.owner}
                  </small>
                </span>
                <Chip>{sourceStatus(d)}</Chip>
              </button>
            ))}</CompactList>
          </section>
        </>
      )}
      {tab === "catalog" && (
        <CompactList className="catalog-grid" label="sources">
          {data?.catalog.map((s) => (
            <a
              className="panel catalog-item"
              key={s.id}
              href={s.url}
              target="_blank"
              rel="noreferrer"
            >
              <ExternalLink size={16} />
              <h3>{s.title}</h3>
              <p>
                {s.authority} · {s.category}
              </p>
              <Chip>Catalog reference</Chip>
            </a>
          ))}
        </CompactList>
      )}
      {tab === "add" && (
        <form className="panel" onSubmit={add}>
          <h2>Add an original reference</h2>
          <p>
            Paste approved source text with its attribution. A clause-library
            category does not itself certify approval.
          </p>
          <label>
            Title
            <input
              required
              maxLength={200}
              value={title}
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
          <label>
            Purpose
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {["policy", "playbook", "clause-library", "reference"].map(
                (c) => (
                  <option key={c}>{c}</option>
                ),
              )}
            </select>
          </label>
          <label>
            Original source URL
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </label>
          <label>
            Source text
            <textarea
              required
              minLength={80}
              maxLength={30000}
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={8}
            />
          </label>
          <button disabled={busy}>Index reference</button>
        </form>
      )}
    </>
  );
}

export function EntitiesPage({ documents }: { documents: Document[] }) {
  const [items, setItems] = useState<Entity[]>([]);
  const [name, setName] = useState("");
  const [type, setType] = useState("vendor");
  const [docId, setDocId] = useState("");
  const [excerpt, setExcerpt] = useState("");
  const [relationship, setRelationship] = useState("mentioned in");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(
    () =>
      workspace
        .entities()
        .then(setItems)
        .catch((e) => setError(message(e))),
    [],
  );
  useEffect(() => {
    let active = true;
    void workspace
      .entities()
      .then((result) => {
        if (active) setItems(result);
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, []);
  async function add(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await workspace.entity({
        name,
        entity_type: type,
        document_id: docId,
        excerpt,
        relationship,
      });
      setName("");
      setExcerpt("");
      await load();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="overline">Connect the facts.</p>
          <h1>Entities & relationships</h1>
          <p>
            Build source-backed records of vendors, agencies, contracts, and
            obligations. Review every proposed connection.
          </p>
        </div>
      </div>
      <ErrorNotice error={error} />
      <div className="two-columns">
        <form className="panel" onSubmit={add}>
          <h2>Propose a sourced entity</h2>
          <p className="muted">
            Exact normalized names are matched; this is not an autonomous
            identity decision.
          </p>
          <label>
            Name
            <input
              required
              maxLength={200}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            Entity type
            <select value={type} onChange={(e) => setType(e.target.value)}>
              {[
                "vendor",
                "agency",
                "organization",
                "obligation",
                "contract",
              ].map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
          <label>
            Source document
            <select
              required
              value={docId}
              onChange={(e) => setDocId(e.target.value)}
            >
              <option value="">Select an indexed document</option>
              {documents.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.title}
                </option>
              ))}
            </select>
          </label>
          <label>
            Supporting excerpt
            <textarea
              required
              value={excerpt}
              maxLength={1500}
              onChange={(e) => setExcerpt(e.target.value)}
              placeholder="Paste the exact supporting passage from the document."
            />
          </label>
          <label>
            Relationship to document
            <input
              required
              maxLength={100}
              value={relationship}
              onChange={(e) => setRelationship(e.target.value)}
            />
          </label>
          <button disabled={busy || !docId}>Propose entity</button>
        </form>
        <section>
          {!items.length && (
            <Empty title="No entity records yet">
              Start with an actual document passage. All new or changed records
              return to draft for review.
            </Empty>
          )}
          <CompactList label="entities">{items.map((item) => (
            <article className="panel" key={item.id}>
              <h2>{item.name}</h2>
              <Chip>{item.entity_type}</Chip>
              {item.links.map((l, i) => (
                <blockquote key={i}>
                  <strong>
                    {l.relationship} →{" "}
                    {documents.find((d) => d.id === l.document_id)?.title ||
                      "Unavailable document"}
                  </strong>
                  <p>{l.excerpt}</p>
                  <small>Source version {l.version}</small>
                </blockquote>
              ))}
              <Decision key={`${item.id}:${item.revision}`} item={item} onChange={() => void load()} />
            </article>
          ))}</CompactList>
        </section>
      </div>
    </>
  );
}

export function RecordsPage({ documents = [], onOpen }: { documents?: Document[]; onOpen?: (doc: Document) => void }) {
  const [items, setItems] = useState<StructuredRecord[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [schemaBusy, setSchemaBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const [records, result] = await Promise.all([
        workspace.records(),
        workspace.reviews(),
      ]);
      setItems(records);
      setReviews(result);
      setError("");
    } catch (e) {
      setError(message(e));
    }
  }, []);
  const all = [...items, ...reviews];
  const matches = (item: StructuredRecord | Review) => {
    const title = "title" in item ? item.title : documents.find(d => d.id === item.document_id)?.title || item.analysis.document_summary;
    return (filter === "all" || (filter === "pending" ? item.decision === "draft"
      : filter === "blocked" ? item.readiness?.can_approve === false : item.readiness?.can_export === true)) &&
      title.toLowerCase().includes(query.trim().toLowerCase());
  };
  useEffect(() => {
    let active = true;
    void Promise.all([workspace.records(), workspace.reviews()])
      .then(([records, result]) => {
        if (active) {
          setItems(records);
          setReviews(result);
        }
      })
      .catch((e) => {
        if (active) setError(message(e));
      });
    return () => {
      active = false;
    };
  }, []);
  return (
    <>
      <div className="page-heading">
        <div>
          <p className="overline">A decision with a record.</p>
          <h1>Reviewed records</h1>
          <p>
            Review generated memos and contract findings before exporting a
            source-linked JSON record.
          </p>
        </div>
        <div className="actions">
        <button className="quiet" disabled={schemaBusy} onClick={() => {
          setSchemaBusy(true); setError("");
          void workspace.exportSchema().then(schema => downloadJson(schema, "kana-legal-export-schema-1.1.json"))
            .catch(error => setError(message(error))).finally(() => setSchemaBusy(false));
        }}><Download size={16} />Export schema</button>
        <button className="quiet" onClick={() => void load()}>
          <RefreshCw size={16} />
          Refresh
        </button>
        </div>
      </div>
      <ErrorNotice error={error} />
      <div className="panel triage-toolbar">
        <div className="filter-buttons" role="group" aria-label="Filter review records">
          {[["pending", "Needs decision", all.filter(r => r.decision === "draft").length],
            ["blocked", "Action needed", all.filter(r => r.readiness?.can_approve === false).length],
            ["exportable", "Ready to export", all.filter(r => r.readiness?.can_export === true).length],
            ["all", "All records", all.length]].map(([key, label, count]) =>
            <button key={key} className="quiet" aria-pressed={filter === key} onClick={() => setFilter(String(key))}>{label} <span className="count">{count}</span></button>
          )}
        </div>
        <label className="search-box"><Search size={16} /><input aria-label="Find a review record" placeholder="Find a contract or memo" value={query} onChange={e => setQuery(e.target.value)} /></label>
      </div>
      {!items.length && !reviews.length && (
        <Empty title="No reviewable outputs yet">
          Run a contract review, or save a cited research answer as a record.
        </Empty>
      )}
      {all.length > 0 && !all.some(matches) && <Empty title="No records in this view">
        Choose All records or clear the search to inspect the rest of your decision history.
      </Empty>}
      <CompactList key={`${filter}:${query}`} className="records-grid" label="records">
        {items.filter(matches).map((item) => (
          <article className="panel" key={item.id}>
            <Chip>Research record</Chip>
            <h2>{item.title}</h2>
            <p className="answer-text">{item.content.answer}</p>
            <details>
              <summary>Evidence and generation mode</summary>
              <p>{item.content.synthesis_mode}</p>
              {item.content.evidence.map((e) => (
                <blockquote key={e.evidence_id}>
                  <strong>{e.title}</strong>
                  <p>{e.excerpt}</p>
                  <small>{e.evidence_id}</small>
                </blockquote>
              ))}
            </details>
            <Decision key={`${item.id}:${item.revision}`} item={item} onChange={() => void load()} />
          </article>
        ))}
        {reviews.filter(matches).map((item) => (
          <article className="panel" key={item.id}>
            <Chip>Contract review</Chip>
            <h2>{documents.find(d => d.id === item.document_id)?.title || "Contract review"}</h2>
            <p>{item.analysis.findings.length} findings · {item.analysis.overall_risk} priority</p>
            <p>{item.analysis.document_summary}</p>
            <small>
              Source version {item.document_version} · {item.analysis.engine}
            </small>
            <details>
              <summary>Inspect findings before decision</summary>
              <FindingList findings={item.analysis.findings} />
            </details>
            {onOpen && documents.some(d => d.id === item.document_id) && <button className="text-button" onClick={() => {
              const doc = documents.find(d => d.id === item.document_id); if (doc) onOpen(doc);
            }}>Open contract and evidence</button>}
            <Decision key={`${item.id}:${item.revision}`} item={item} onChange={() => void load()} />
          </article>
        ))}
      </CompactList>
    </>
  );
}
