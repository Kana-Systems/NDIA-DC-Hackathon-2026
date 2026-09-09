import { useState } from "react";
import { LoaderCircle, ShieldCheck } from "lucide-react";
import { apiClient } from "./api/client";
import { ErrorNotice } from "./WorkspaceShared";
import { message } from "./workspaceUtils";

export function SessionRenewal({ onRenewed }: { onRenewed: () => void }) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  return <div className="session-overlay" role="dialog" aria-modal="true" aria-labelledby="session-renewal-title" aria-describedby="session-renewal-description">
    <form className="panel" onKeyDown={event => {
      if (event.key !== "Tab") return;
      const controls = event.currentTarget.querySelectorAll<HTMLElement>("input:not(:disabled), button:not(:disabled)");
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && event.target === first) { event.preventDefault(); last?.focus(); }
      if (!event.shiftKey && event.target === last) { event.preventDefault(); first?.focus(); }
    }} onSubmit={async event => {
      event.preventDefault();
      if (busy) return;
      setBusy(true); setError("");
      try {
        const result = await apiClient.login(password);
        if (result.demoMode || !result.data.authenticated) throw new Error("A connected workspace session is required.");
        onRenewed();
      } catch (error) { setError(message(error)); }
      finally { setBusy(false); }
    }}>
      <ShieldCheck size={28} aria-hidden="true" />
      <h2 id="session-renewal-title">Resume your workspace</h2>
      <p id="session-renewal-description">Your session expired. Your open forms are still here. Sign in, then retry the action you were taking.</p>
      <label>Workspace password<input autoFocus type="password" autoComplete="current-password" required value={password} onChange={event => setPassword(event.target.value)} /></label>
      <ErrorNotice error={error} />
      <button disabled={busy || !password}>{busy && <LoaderCircle className="spin" size={16} />}Resume workspace</button>
    </form>
  </div>;
}
