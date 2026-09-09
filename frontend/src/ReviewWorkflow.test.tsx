import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import { FindingList } from "./FindingList";
import { Decision, WorkspaceTabs } from "./WorkspaceShared";
import { Research } from "./ContractWorkspace";
import { workspace } from "./workspaceApi";
import type { Review } from "./workspaceApi";

const finding = { id: "low", title: "Routine provision", category: "Warranty", severity: "low", confidence: null,
  excerpt: "The warranty lasts twelve months.", explanation: "Check the warranty period.", recommendation: "Confirm duration.",
  grounding_status: "verified", citations: [{ title: "Playbook", url: "https://example.org", section: "Warranty", excerpt: "Warranty", verification_status: "retrieved_evidence" }],
} satisfies Review["analysis"]["findings"][number];
const findings = [finding, { ...finding, id: "high", title: "Urgent obligation", severity: "high" as const, grounding_status: "unverified", citations: [] }];

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("Review triage and decision readiness", () => {
  it("prioritizes high findings and combines evidence filters with search", async () => {
    const user = userEvent.setup();
    render(<FindingList findings={findings} />);
    expect(screen.getAllByRole("heading", { level: 2 }).map(h => h.textContent)).toEqual(["Urgent obligation", "Routine provision"]);
    await user.click(screen.getByRole("button", { name: /needs evidence/i }));
    expect(screen.getByRole("heading", { name: "Urgent obligation" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Routine provision" })).not.toBeInTheDocument();
    expect(screen.getByText("Evidence needs review")).toBeInTheDocument();
    await user.type(screen.getByRole("textbox", { name: "Search findings" }), "absent");
    expect(screen.getByRole("status")).toHaveTextContent("Showing 0 of 2");
    expect(screen.getByRole("heading", { name: "No findings match these filters" })).toBeInTheDocument();
  });

  it("keeps stale approval and export disabled while allowing a return for revision", async () => {
    const user = userEvent.setup();
    const decide = vi.spyOn(workspace, "decide").mockResolvedValue({});
    const onChange = vi.fn();
    render(<Decision item={{ id: "review", kind: "review", revision: 2, owner: "alice", security_domain: "demo", created_at: "", updated_at: "",
      decision: "approved", note: "Checked original sources", readiness: { can_approve: false, can_export: false, checked_at: "", blockers: [{ code: "source_changed", message: "Source changed. Create a fresh draft." }] },
    }} onChange={onChange} />);
    expect(screen.getByText("Source changed. Create a fresh draft.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Approve output" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Export JSON" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Return for revision" }));
    expect(decide).toHaveBeenCalledWith(expect.objectContaining({ id: "review", revision: 2 }), "rejected", "Checked original sources");
    expect(onChange).toHaveBeenCalledOnce();
  });

  it("supports manual keyboard tab activation with labelled panels", async () => {
    function Example() {
      const [value, setValue] = useState("first");
      return <WorkspaceTabs label="Review" tabs={[["first", "Findings"], ["second", "Document"], ["third", "History"]]} value={value} onChange={setValue}>
        <p>{value} content</p>
      </WorkspaceTabs>;
    }
    const user = userEvent.setup();
    render(<Example />);
    screen.getByRole("tab", { name: "Findings" }).focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Document" })).toHaveFocus();
    expect(screen.getByRole("tab", { name: "Findings" })).toHaveAttribute("aria-selected", "true");
    await user.keyboard("{Enter}");
    const panel = screen.getByRole("tabpanel", { name: "Document" });
    expect(panel).toHaveTextContent("second content");
    expect(screen.getByRole("tab", { name: "Document" })).toHaveAttribute("aria-controls", panel.id);
    await user.keyboard("{End}{Enter}");
    expect(screen.getByRole("tabpanel", { name: "History" })).toBeInTheDocument();
    await user.keyboard("{ArrowRight}{Enter}");
    expect(screen.getByRole("tabpanel", { name: "Findings" })).toBeInTheDocument();
  });
});

describe("Grounded research", () => {
  it("sets a reviewable prompt without starting generation or spending inference", async () => {
    vi.spyOn(workspace, "questions").mockResolvedValue([]);
    const ask = vi.spyOn(workspace, "ask");
    const user = userEvent.setup();
    render(<Research documentId={null} documents={[]} />);
    await user.click(within(screen.getByRole("group", { name: "Research starting points" })).getByRole("button", { name: "Draft a decision memo" }));
    expect(screen.getByLabelText("Output")).toHaveValue("draft");
    expect((screen.getByRole("textbox", { name: "Question or drafting request" }) as HTMLTextAreaElement).value).toContain("mission question");
    expect(screen.getByRole("textbox", { name: "Question or drafting request" })).toHaveFocus();
    expect(ask).not.toHaveBeenCalled();
  });
});
