import { request } from "./api/client";
import type {
  AcquisitionMetadata,
  IntelligenceResponse,
  NativeReport,
  RiskLevel,
} from "./types";
export interface RecordBase {
  id: string;
  kind: string;
  revision: number;
  updated_at: string;
  created_at: string;
  owner: string;
  security_domain: string;
}
export interface Document extends RecordBase {
  title: string;
  text?: string;
  category: string;
  version: string;
  metadata: AcquisitionMetadata | null;
  connection_id?: string;
  source_key?: string;
  source_url?: string;
  status: string;
}
export interface Review extends RecordBase {
  document_id: string;
  document_version: string;
  decision: string;
  note: string;
  decided_by?: string;
  analysis: {
    document_summary: string;
    overall_risk: RiskLevel;
    confidence: null;
    engine: string;
    disclaimer: string;
    findings: {
      id: string;
      title: string;
      category: string;
      severity: RiskLevel;
      confidence: number | null;
      excerpt: string;
      explanation: string;
      recommendation: string;
      grounding_status: string;
      citations: {
        title: string;
        url: string | null;
        excerpt: string;
        section: string;
        verification_status: string;
      }[];
    }[];
    report: NativeReport;
  };
}
export interface Question extends RecordBase {
  query: string;
  document_id: string | null;
  response: IntelligenceResponse;
}
export interface Connection extends RecordBase {
  name: string;
  provider: string;
  folder: string;
  category: string;
  status: string;
  last_sync: string | null;
  counts: Record<string, number>;
  errors: { file: string; message: string }[];
}
export interface Connections {
  search?: string;
  document_storage?: string;
  connections: Connection[];
  shared_folder_available: boolean;
  sharepoint_available: boolean;
  automatic_sync_seconds: number;
  persistence: string;
  security_domain: string;
}
export interface Entity extends RecordBase {
  name: string;
  entity_type: string;
  decision: string;
  note: string;
  method: string;
  links: {
    document_id: string;
    version: string;
    excerpt: string;
    relationship: string;
  }[];
}
export interface StructuredRecord extends RecordBase {
  title: string;
  document_id: string | null;
  decision: string;
  note: string;
  content: IntelligenceResponse;
}
export interface AuditEvent extends RecordBase {
  action: string;
  record_id: string;
  details: Record<string, unknown>;
}
export interface Library {
  corpus: { documents: number; chunks: number; retrieved_at: string } | null;
  evidence: IntelligenceResponse["evidence"];
  catalog: {
    id: string;
    title: string;
    url: string;
    authority: string;
    category: string;
  }[];
  documents: Document[];
}
const call = <T>(path: string, method = "GET", body?: unknown) =>
  request<T>(
    `/api/workspace${path}`,
    { method, ...(body === undefined ? {} : { body: JSON.stringify(body) }) },
    true,
  );
export const workspace = {
  documents: () => call<Document[]>("/documents"),
  document: (id: string) => call<Document>(`/documents/${id}`),
  create: (body: {
    title: string;
    text: string;
    category: string;
    source_url?: string;
  }) => call<Document>("/documents", "POST", body),
  upload: (file: File, category: string) => {
    const body = new FormData();
    body.append("file", file);
    body.append("category", category);
    return request<Document>(
      "/api/workspace/documents/upload",
      { method: "POST", body },
      true,
    );
  },
  metadata: (id: string, metadata: AcquisitionMetadata, revision: number) =>
    call<Document>(`/documents/${id}/metadata`, "PUT", { metadata, revision }),
  review: (id: string) => call<Review>(`/documents/${id}/review`, "POST"),
  reviews: () => call<Review[]>("/reviews"),
  decide: <T>(item: RecordBase, decision: string, note: string) =>
    call<T>(`/records/${item.id}/decision`, "POST", {
      decision,
      note,
      revision: item.revision,
    }),
  export: (id: string) =>
    call<Record<string, unknown>>(`/records/${id}/export`),
  ask: (
    query: string,
    mode: string,
    document_id: string | null,
    finding_id?: string,
  ) =>
    call<Question>("/questions", "POST", {
      query,
      mode,
      document_id,
      finding_id,
    }),
  questions: () => call<Question[]>("/questions"),
  connections: () => call<Connections>("/connections"),
  checkSharePoint: () => call<{site: string; library: string; url: string; status: string}>(
    "/connections/sharepoint/check", "POST"),
  reindex: (id: string) => call<Document>(`/documents/${id}/reindex`, "POST"),
  connect: (body: {
    name: string;
    provider: string;
    folder: string;
    category: string;
  }) => call<Connection>("/connections", "POST", body),
  sync: (id: string) => call<Connection>(`/connections/${id}/sync`, "POST"),
  entities: () => call<Entity[]>("/entities"),
  entity: (body: {
    name: string;
    entity_type: string;
    document_id: string;
    excerpt: string;
    relationship: string;
  }) => call<Entity>("/entities", "POST", body),
  records: () => call<StructuredRecord[]>("/structured-records"),
  record: (question_id: string, title: string) =>
    call<StructuredRecord>("/structured-records", "POST", {
      question_id,
      title,
    }),
  events: () => call<AuditEvent[]>("/events"),
  library: (q = "") => call<Library>(`/library?q=${encodeURIComponent(q)}`),
};
