import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  ChevronRight,
  ChevronLeft,
  Database,
  FileText,
  Layers,
  LoaderCircle,
  LogOut,
  MessageSquare,
  Plus,
  Search,
  ShieldCheck,
  Link2,
} from "lucide-react";
import { apiClient, onSessionExpired } from "./api/client";
import { SessionRenewal } from "./SessionRenewal";
import { workspace } from "./workspaceApi";
import type { Document } from "./workspaceApi";
import { Chip, CompactList, Empty, ErrorNotice } from "./WorkspaceShared";
import { message, sourceStatus } from "./workspaceUtils";
import { ContractWorkspace, Research } from "./ContractWorkspace";
import {
  ConnectionsPage,
  EntitiesPage,
  LibraryPage,
  RecordsPage,
} from "./WorkspacePages";
import "./workspace.css";
import kanaLogo from "./assets/kana-systems-logo.png";
type Page =
  | "contracts"
  | "research"
  | "connections"
  | "entities"
  | "library"
  | "records";
const pages = [
  { id: "contracts", label: "Contracts", icon: FileText },
  { id: "research", label: "Ask / Research", icon: MessageSquare },
  { id: "connections", label: "Data connections", icon: Database },
  { id: "library", label: "Source library", icon: BookOpen },
  { id: "entities", label: "Entities & relationships", icon: Link2 },
  { id: "records", label: "Reviewed records", icon: Layers },
] as const;
const navigation = [
  { id: "contracts", label: "Contracts", icon: FileText, children: ["contracts"] },
  { id: "research", label: "Ask / Research", icon: MessageSquare, children: ["research"] },
  { id: "library", label: "Sources", icon: Database, children: ["library", "connections"] },
  { id: "records", label: "Knowledge", icon: Layers, children: ["records", "entities"] },
] as const;
function Login({ onLogin }: { onLogin: () => void }) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function login(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const result = await apiClient.login(password);
      if (result.demoMode || !result.data.authenticated)
        throw new Error("The connected workspace requires the live API.");
      onLogin();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="lens-login">
      <section>
        <span className="wordmark">
          <img className="kana-logo" src={kanaLogo} alt="Kana Systems" width="64" height="64" />
          <span>Kana Legal</span>
        </span>
        <p className="overline">Contract intelligence, connected.</p>
        <h1>
          From source
          <br />
          <span className="hero-accent">to sound judgment.</span>
        </h1>
        <p>
          One workspace for your documents, contract reviews, evidence, and
          decisions.
        </p>
        <div className="login-route">
          <span>Connect</span>
          <ChevronRight />
          <span>Review</span>
          <ChevronRight />
          <span>Decide</span>
        </div>
      </section>
      <form onSubmit={login}>
        <ShieldCheck size={26} />
        <h2>Open your workspace</h2>
        <p>
          For approved, public documents. Your workspace records stay on the
          server, scoped to your identity.
        </p>
        <label>
          Workspace password
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoFocus
          />
        </label>
        <ErrorNotice error={error} />
        <button disabled={busy || !password}>
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <ArrowRight size={16} />
          )}
          Enter workspace
        </button>
      </form>
    </main>
  );
}
function AddDocument({ onAdded }: { onAdded: (doc: Document) => void }) {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (!file && text.trim().length < 80)
        throw new Error("Add at least 80 characters of contract text.");
      const doc = file
        ? await workspace.upload(file, "contract")
        : await workspace.create({ title, text, category: "contract" });
      onAdded(doc);
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="panel add-document" onSubmit={submit}>
      <h2>Add a contract</h2>
      <p>
        Upload an original PDF/DOCX or paste contract text. No sample document
        is inserted.
      </p>
      <label>
        Contract file
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={(e) => setFile(e.target.files?.[0])}
        />
      </label>
      {!file && (
        <>
          <label>
            Document title
            <input
              value={title}
              maxLength={200}
              required
              onChange={(e) => setTitle(e.target.value)}
            />
          </label>
          <label>
            Contract text
            <textarea
              value={text}
              maxLength={30000}
              onChange={(e) => setText(e.target.value)}
              rows={7}
            />
          </label>
        </>
      )}
      <p className="muted">
        Maximum 15 MB per file and 30,000 extracted characters per document
        section.
      </p>
      <ErrorNotice error={error} />
      <button disabled={busy}>
        {busy && <LoaderCircle className="spin" size={16} />}Save to workspace
      </button>
    </form>
  );
}
export default function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [sessionExpired, setSessionExpired] = useState(false);
  const [page, setPage] = useState<Page>("contracts");
  const [collapsed, setCollapsed] = useState(false);
  const mainRef = useRef<HTMLElement>(null);
  const activeGroup = navigation.find(group => (group.children as readonly Page[]).includes(page))!;
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selected, setSelected] = useState<Document | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [query, setQuery] = useState("");
  const [researchDoc, setResearchDoc] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const documentRequest = useRef(0);
  useEffect(() => onSessionExpired(() => setSessionExpired(true)), []);
  const loadDocuments = useCallback(() => {
    const request = ++documentRequest.current;
    return workspace.documents().then(result => {
      if (request === documentRequest.current) { setDocuments(result); setError(""); }
    }).catch(e => {
      if (request === documentRequest.current) setError(message(e));
    }).finally(() => {
      if (request === documentRequest.current) setLoading(false);
    });
  }, []);
  const reload = useCallback(() => {
    setLoading(true);
    return loadDocuments();
  }, [loadDocuments]);
  useEffect(() => {
    if (!authenticated) return;
    void loadDocuments();
    return () => { documentRequest.current += 1; };
  }, [authenticated, loadDocuments]);
  useEffect(() => {
    if (!authenticated || sessionExpired) return;
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [page, authenticated, sessionExpired]);
  if (!authenticated) return <Login onLogin={() => { setLoading(true); setAuthenticated(true); }} />;
  const contracts = documents.filter(
    (d) =>
      d.category === "contract" &&
      d.title.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
    {sessionExpired && <SessionRenewal onRenewed={() => setSessionExpired(false)} />}
    <div inert={sessionExpired} className={`lens-workspace${collapsed ? " sidebar-collapsed" : ""}`}>
      <a className="skip-link" href="#workspace-main">
        Skip to workspace
      </a>
      <aside className="sidebar">
        <div className="sidebar-header">
          <a
            href="#"
            className="wordmark"
            aria-label="Kana Legal home"
            onClick={(e) => {
              e.preventDefault();
              setPage("contracts");
            }}
          >
            <img className="kana-logo" src={kanaLogo} alt="Kana Systems" width="40" height="40" />
            <span className="brand-name">Kana <b>Legal</b></span>
          </a>
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav id="workspace-navigation" aria-label="Main navigation">
          {navigation.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              aria-label={label}
              title={collapsed ? label : undefined}
              aria-current={activeGroup.id === id ? "page" : undefined}
              className={activeGroup.id === id ? "active" : ""}
              onClick={() => {
                if (activeGroup.id !== id) setPage(id);
                void reload();
              }}
            >
              <Icon size={18} aria-hidden="true" />
              <span className="nav-text">{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-controls">
          <button
            className="sidebar-toggle"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-expanded={!collapsed}
            aria-controls="workspace-navigation"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            onClick={() => setCollapsed(value => !value)}
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>
        <div className="sidebar-bottom">
          <ShieldCheck size={18} />
          <p>
            Evidence-led.
            <br />
            Human-reviewed.
          </p>
          <button
            className="quiet"
            aria-label="Sign out"
            title={collapsed ? "Sign out" : undefined}
            onClick={() => {
              apiClient.logout();
              setAuthenticated(false);
              setDocuments([]);
              setSelected(null);
              setPage("contracts");
              setResearchDoc("");
              setShowAdd(false);
              setError("");
            }}
          >
            <LogOut size={15} />
            <span className="nav-text">Sign out</span>
          </button>
        </div>
      </aside>
      <div className="workspace-body">
        <main id="workspace-main" ref={mainRef} tabIndex={-1}>
          {activeGroup.children.length > 1 && (
            <nav className="section-navigation" aria-label={`${activeGroup.label} navigation`}>
              {activeGroup.children.map(id => (
                <button key={id} aria-current={page === id ? "page" : undefined}
                  className={page === id ? "active" : ""} onClick={() => setPage(id)}>
                  {pages.find(item => item.id === id)?.label}
                </button>
              ))}
            </nav>
          )}
          <ErrorNotice error={error} />
          {page === "contracts" &&
            (selected ? (
              <ContractWorkspace
                key={selected.id}
                initial={selected}
                documents={documents}
                onBack={() => {
                  setSelected(null);
                  void reload();
                }}
                onChanged={() => void reload()}
              />
            ) : (
              <>
                <div className="page-heading">
                  <div>
                    <p className="overline">Start with the source.</p>
                    <h1>Your contract workspace</h1>
                    <p>
                      Review agreements, follow the evidence, and keep a record
                      of every decision.
                    </p>
                  </div>
                  <button onClick={() => setShowAdd(!showAdd)}>
                    <Plus size={17} />
                    {showAdd ? "Close form" : "Add contract"}
                  </button>
                </div>
                <div className="journey">
                  <article>
                    <span>01</span>
                    <h3>Connect your sources</h3>
                    <p>Ingest approved folders and track source changes.</p>
                    <button
                      className="text-button"
                      onClick={() => setPage("connections")}
                    >
                      Manage sources <ArrowRight size={14} />
                    </button>
                  </article>
                  <article>
                    <span>02</span>
                    <h3>Research with evidence</h3>
                    <p>Answer a mission question or draft a cited review memo.</p>
                    <button className="text-button" onClick={() => setPage("research")}>
                      Start research <ArrowRight size={14} />
                    </button>
                  </article>
                  <article>
                    <span>03</span>
                    <h3>Review and share</h3>
                    <p>Resolve blockers, record a decision, and export sourced results.</p>
                    <button
                      className="text-button"
                      onClick={() => setPage("records")}
                    >
                      Reviewed records <ArrowRight size={14} />
                    </button>
                  </article>
                </div>
                {showAdd && (
                  <AddDocument
                    onAdded={(doc) => {
                      setShowAdd(false);
                      setSelected(doc);
                      void reload();
                    }}
                  />
                )}
                <div className="section-heading">
                  <h2>
                    Contracts <span className="count">{contracts.length}</span>
                  </h2>
                  <label className="search-box">
                    <Search size={16} />
                    <input
                      aria-label="Find a contract"
                      placeholder="Find a contract"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                  </label>
                </div>
                {loading && <p role="status">Loading saved documents…</p>}
                {!loading && !contracts.length && (
                  <Empty
                    title={
                      query
                        ? "No matching contracts"
                        : "Your next review starts here"
                    }
                  >
                    {query
                      ? "Try another title."
                      : "Add an original contract or connect an approved shared folder. No demonstration records are inserted."}
                  </Empty>
                )}
                <CompactList key={query} className="contract-list" label="contracts">
                  {contracts.map((doc) => (
                    <button
                      key={doc.id}
                      className="contract-row"
                      disabled={doc.status === "source-error"}
                      onClick={() => setSelected(doc)}
                    >
                      <span className="document-icon">
                        <FileText size={22} />
                      </span>
                      <span>
                        <strong>{doc.title}</strong>
                        <small>
                          {doc.metadata?.agency || "Acquisition details needed"}{" "}
                          · {new Date(doc.updated_at).toLocaleDateString()}
                        </small>
                      </span>
                      <Chip>{sourceStatus(doc)}</Chip>
                      <ChevronRight size={18} />
                    </button>
                  ))}
                </CompactList>
              </>
            ))}
          {page === "research" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="overline">Research with receipts.</p>
                  <h1>Ask / Research</h1>
                  <p>
                    Ground an answer or a review memo in the documents you can
                    access.
                  </p>
                </div>
              </div>
              <label className="context-picker">
                Research context
                <select
                  value={researchDoc}
                  onChange={(e) => setResearchDoc(e.target.value)}
                >
                  <option value="">All indexed sources</option>
                  {documents
                    .filter((d) => d.available !== false && d.status === "ready")
                    .map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.title}
                      </option>
                    ))}
                </select>
              </label>
              <Research
                key={researchDoc}
                documentId={researchDoc || null}
                documents={documents}
              />
            </>
          )}
          {page === "connections" && (
            <ConnectionsPage onChanged={() => void reload()} />
          )}
          {page === "library" && (
            <LibraryPage
              onOpen={(doc) => {
                setSelected(doc);
                setPage("contracts");
              }}
            />
          )}
          {page === "entities" && <EntitiesPage documents={documents} />}
          {page === "records" && <RecordsPage documents={documents} onOpen={doc => {
            setSelected(doc); setPage("contracts");
          }} />}
        </main>
        <footer className="workspace-footer">
          <ShieldCheck size={14} />
          Human review required. Screening and drafts are not legal advice or
          compliance certification.
        </footer>
      </div>
    </div>
    </>
  );
}
