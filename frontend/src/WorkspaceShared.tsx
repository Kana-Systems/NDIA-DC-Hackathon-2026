import { Children, useId, useRef, useState } from "react";
import { Check, Download, FileText } from "lucide-react";
import { workspace } from "./workspaceApi";
import type { RecordBase } from "./workspaceApi";
import { downloadJson, message } from "./workspaceUtils";

export function CompactList({ children, label, className }: {
  children: React.ReactNode;
  label: string;
  className?: string;
}) {
  const rows = Children.toArray(children);
  const [expanded, setExpanded] = useState(false);
  const id = useId();
  const remaining = rows.length - 10;
  return <div className="compact-list">
    <div className={className}>{rows.slice(0, 10)}</div>
    {remaining > 0 && <>
      <button type="button" className="quiet compact-list-toggle" aria-expanded={expanded}
        aria-controls={id} onClick={() => setExpanded(!expanded)}>
        {expanded ? `Show fewer ${label}` : `Show ${remaining} more ${label}`}
      </button>
      {/* Keep hidden rows mounted so collapsing a list cannot discard decision notes. */}
      <div id={id} className="compact-list-overflow" role="region" aria-label={`Additional ${label}`}
        tabIndex={0} hidden={!expanded}>
        <div className={className}>{rows.slice(10)}</div>
      </div>
    </>}
  </div>;
}

export function ErrorNotice({ error }: { error: string }) {
  return error ? (
    <p className="notice error" role="alert">
      {error}
    </p>
  ) : null;
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="workspace-empty">
      <FileText size={28} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Chip({ children }: { children: React.ReactNode }) {
  return <span className="chip">{children}</span>;
}
export function Decision({
  item,
  onChange,
}: {
  item: RecordBase & { decision: string; note: string };
  onChange: () => void;
}) {
  const [note, setNote] = useState(item.note || "");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const readiness = item.readiness;
  async function act(decision: string) {
    setBusy(true);
    setError("");
    try {
      await workspace.decide(item, decision, note);
      onChange();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="decision">
      <div className="section-heading">
        <h3>Human review</h3>
        <Chip>{item.decision}</Chip>
      </div>
      {item.decided_by && <p className="decision-attribution">Last decision by {item.decided_by}
        {item.decided_at && <> · <time dateTime={item.decided_at}>{new Date(item.decided_at).toLocaleString()}</time></>}
      </p>}
      <p>
        Check the source passages and recommendations, then record your judgment.
        Approval records a human review of this output.
      </p>
      {readiness && (
        <div className={`readiness-status ${readiness.can_approve ? "ready" : "blocked"}`}>
          <strong>{readiness.can_export ? "Ready to export" : readiness.can_approve ? "Ready for your decision" : "Action needed before approval"}</strong>
          {readiness.blockers.map(blocker => <p key={blocker.code}>{blocker.message}</p>)}
        </div>
      )}
      <label>
        Decision note
        <textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          minLength={3}
          maxLength={2000}
          placeholder="Record what you checked and any remaining limitations."
        />
      </label>
      <ErrorNotice error={error} />
      <div className="actions">
        <button
          disabled={busy || note.trim().length < 3 || readiness?.can_approve === false}
          onClick={() => void act("approved")}
        >
          <Check size={16} />
          Approve output
        </button>
        <button
          className="quiet"
          disabled={busy || note.trim().length < 3}
          onClick={() => void act("rejected")}
        >
          Return for revision
        </button>
        {item.decision === "approved" && item.kind !== "entity" && (
          <button
            className="quiet"
            disabled={busy || readiness?.can_export === false}
            onClick={() => {
              setBusy(true);
              void workspace
                .export(item.id)
                .then((data) => downloadJson(data, `kana-legal-${item.id}.json`))
                .catch((e) => setError(message(e)))
                .finally(() => setBusy(false));
            }}
          >
            <Download size={16} />
            Export JSON
          </button>
        )}
      </div>
    </div>
  );
}

/** Manual activation preserves drafts while arrow keys move among tabs. */
export function WorkspaceTabs({ label, tabs, value, onChange, children }: {
  label: string;
  tabs: readonly (readonly [string, string])[];
  value: string;
  onChange: (value: string) => void;
  children: React.ReactNode;
}) {
  const id = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  return <>
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map(([key, title], index) => <button
        key={key} type="button" role="tab" id={`${id}-tab-${key}`}
        aria-controls={`${id}-panel-${key}`} aria-selected={value === key}
        tabIndex={value === key ? 0 : -1} className={value === key ? "active" : ""}
        ref={element => { refs.current[index] = element; }}
        onClick={() => onChange(key)}
        onKeyDown={event => {
          const next = event.key === "ArrowRight" ? (index + 1) % tabs.length
            : event.key === "ArrowLeft" ? (index - 1 + tabs.length) % tabs.length
            : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
          if (next !== null) { event.preventDefault(); refs.current[next]?.focus(); }
        }}
      >{title}</button>)}
    </div>
    {tabs.map(([key]) => <div key={key} role="tabpanel" id={`${id}-panel-${key}`} aria-labelledby={`${id}-tab-${key}`} hidden={value !== key} tabIndex={0}>
      {value === key ? children : null}
    </div>)}
  </>;
}
