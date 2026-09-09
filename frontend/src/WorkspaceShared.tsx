import { useState } from "react";
import { Check, Download, FileText } from "lucide-react";
import { workspace } from "./workspaceApi";
import type { RecordBase } from "./workspaceApi";
import { downloadJson, message } from "./workspaceUtils";
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
      <p>
        Approval records your review of this output—not a determination of legal
        compliance.
      </p>
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
          disabled={busy || note.trim().length < 3}
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
            disabled={busy}
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
