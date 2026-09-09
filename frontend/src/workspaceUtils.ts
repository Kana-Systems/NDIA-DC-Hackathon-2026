export function downloadJson(data: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export const message = (error: unknown) =>
  error instanceof Error ? error.message : "The request could not be completed.";

export function sourceStatus(doc: { status: string; available?: boolean }) {
  if (doc.status === "source-error") return "Source unavailable";
  if (doc.status === "index-failed") return "Indexing failed";
  if (doc.status === "indexing") return "Indexing";
  if (doc.available === false) return "Sync required";
  return doc.status === "ready" ? "Ready" : doc.status;
}
