"""Connected Lens workflow: documents, ingestion, reviews, cited questions and exports."""

import asyncio
import hashlib
import json
import re
from copy import copy
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field

from app.api import get_review_service
from app.config import Settings, get_settings
from app.intelligence import CitedGenerationService
from app.lens import present_report, require_reviewer
from app.local_retrieval import FederalCorpusRetrieval
from app.models import (
    AcquisitionMetadata,
    DocumentLocation,
    Evidence,
    ExtractedSegment,
    IntelligenceQuery,
    ParsedDocument,
    PrincipalContext,
)
from app.parsers import DocumentParser
from app.workspace_store import now

router = APIRouter(prefix="/api/workspace", tags=["lens-workspace"])
Principal = Annotated[PrincipalContext, Depends(require_reviewer)]
Configuration = Annotated[Settings, Depends(get_settings)]


def store(request):
    return request.app.state.workspace_store


def document_record(request, principal, record_id):
    item = store(request).get(principal, record_id)
    if item["kind"] != "document" or item.get("deleted") or item.get("status") == "source-error":
        raise HTTPException(404, "Document unavailable")
    return item


def parsed_text(title, text):
    paragraphs = [text[i : i + 1800] for i in range(0, len(text), 1800)]
    return ParsedDocument(
        filename=title,
        media_type="text/plain",
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        segments=[
            ExtractedSegment(
                segment_id=f"text-{i}",
                text=part,
                location=DocumentLocation(paragraph=i, label=f"Text block {i}"),
            )
            for i, part in enumerate(paragraphs, 1)
        ],
    )


class DocumentInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=80, max_length=30000)
    category: Literal["contract", "policy", "playbook", "clause-library", "reference"] = "contract"
    metadata: AcquisitionMetadata | None = None
    source_url: str = Field(default="", max_length=2000, pattern=r"^(https?://.*)?$")


def save_document(db, principal, document, category, metadata=None, **extra):
    if len(document.text) > 30000:
        raise HTTPException(413, "Split this document into sections of at most 30,000 characters.")
    record_id = extra.pop("record_id", None)
    payload = {
        "title": document.filename,
        "text": document.text,
        "sha256": document.sha256,
        "parsed": document.model_dump(mode="json"),
        "category": category,
        "metadata": metadata,
        "status": "ready",
        "deleted": False,
        "version": document.sha256[:12],
        **extra,
    }
    item = db.save(principal, "document", payload, record_id)
    db.event(principal, "document.indexed", item["id"], {"version": item["version"]})
    return item


@router.get("/documents")
def documents(request: Request, principal: Principal):
    return [
        {k: v for k, v in item.items() if k not in {"text", "parsed"}}
        for item in store(request).list(principal, "document")
        if not item.get("deleted")
    ]


@router.post("/documents")
def create_document(payload: DocumentInput, request: Request, principal: Principal):
    return save_document(
        store(request),
        principal,
        parsed_text(payload.title, payload.text),
        payload.category,
        payload.metadata.model_dump(mode="json") if payload.metadata else None,
        source_url=payload.source_url,
    )


@router.post("/documents/upload")
async def upload_document(
    request: Request,
    principal: Principal,
    settings: Configuration,
    file: Annotated[UploadFile, File()],
    category: Annotated[str, Form()] = "contract",
):
    if category not in {"contract", "policy", "playbook", "clause-library", "reference"}:
        raise HTTPException(422, "Invalid document category")
    parser = DocumentParser(settings)
    data = await parser.read_upload(file)
    try:
        document = await asyncio.to_thread(parser.parse, file.filename or "upload", data)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return save_document(store(request), principal, document, category)


@router.get("/documents/{record_id}")
def get_document(record_id: str, request: Request, principal: Principal):
    return document_record(request, principal, record_id)


class MetadataInput(BaseModel):
    metadata: AcquisitionMetadata
    revision: int


@router.put("/documents/{record_id}/metadata")
def metadata(record_id: str, payload: MetadataInput, request: Request, principal: Principal):
    item = document_record(request, principal, record_id)
    item["metadata"] = payload.metadata.model_dump(mode="json")
    item["status"] = "ready"
    return store(request).save(principal, "document", item, record_id, payload.revision)


@router.post("/documents/{record_id}/review")
async def review(record_id: str, request: Request, principal: Principal, settings: Configuration):
    item = document_record(request, principal, record_id)
    if item["category"] != "contract" or not item.get("metadata"):
        raise HTTPException(422, "Select a contract and save its acquisition details first.")
    service = copy(get_review_service(request))
    if hasattr(service, "llm"):
        service.llm = copy(service.llm)
    if getattr(service, "retrieval", None) is not None:
        service.retrieval = WorkspaceRetrieval(
            store(request), principal, settings, references_only=True
        )
    async with request.app.state.review_semaphore:
        try:
            report = await asyncio.to_thread(
                service.review,
                ParsedDocument.model_validate(item["parsed"]),
                AcquisitionMetadata.model_validate(item["metadata"]),
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            raise HTTPException(
                503, "Review failed. Check model, corpus and Bedrock access."
            ) from error
    result = store(request).save(
        principal,
        "review",
        {
            "document_id": record_id,
            "document_version": item["version"],
            "metadata_snapshot": item["metadata"],
            "analysis": present_report(report),
            "decision": "draft",
            "note": "",
            "decided_by": None,
        },
    )
    store(request).event(
        principal,
        "review.completed",
        result["id"],
        {"document_id": record_id, "engine": report.synthesis_mode},
    )
    return result


@router.get("/reviews")
def reviews(request: Request, principal: Principal):
    return sorted(
        store(request).list(principal, "review"), key=lambda r: r["created_at"], reverse=True
    )


class DecisionInput(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = Field(min_length=3, max_length=2000)
    revision: int


@router.post("/records/{record_id}/decision")
def decide(record_id: str, payload: DecisionInput, request: Request, principal: Principal):
    item = store(request).get(principal, record_id)
    if item["kind"] not in {"review", "entity", "structured-record"}:
        raise HTTPException(422, "This record cannot be approved")
    if payload.decision == "approved":
        ensure_current_sources(item, request, principal)
    if item["kind"] == "review":
        document = document_record(request, principal, item["document_id"])
        if (
            document["version"] != item["document_version"]
            or document["metadata"] != item["metadata_snapshot"]
        ):
            raise HTTPException(409, "Contract or acquisition details changed. Run a new review.")
    item.update(
        decision=payload.decision, note=payload.note, decided_by=principal.subject, decided_at=now()
    )
    result = store(request).save(principal, item["kind"], item, record_id, payload.revision)
    store(request).event(principal, "analyst." + payload.decision, record_id, {})
    return result


@router.get("/records/{record_id}/export")
def export(record_id: str, request: Request, principal: Principal):
    item = store(request).get(principal, record_id)
    if item["kind"] not in {"review", "structured-record"} or item.get("decision") != "approved":
        raise HTTPException(409, "Approve the record before exporting.")
    ensure_current_sources(item, request, principal)
    if item["kind"] == "review":
        document = document_record(request, principal, item["document_id"])
        if (
            document["version"] != item["document_version"]
            or document["metadata"] != item["metadata_snapshot"]
        ):
            raise HTTPException(409, "Review is stale. Re-review the updated contract.")
    store(request).event(principal, "record.exported", record_id, {})
    return {
        "schema_version": "1.0",
        "exported_at": now(),
        "external_write_performed": False,
        "disclaimer": "Analyst-reviewed screening output; not a compliance certification.",
        "record": item,
    }


class QuestionInput(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    mode: Literal["answer", "summary", "draft"] = "answer"
    document_id: str | None = None
    finding_id: str | None = None


class WorkspaceGeneration(CitedGenerationService):
    """Live LLM errors are visible; explicit offline mode quotes actual evidence."""

    def _synthesize(self, request, evidence):
        if self.settings.bedrock_enabled:
            return self._remote_synthesis(request, evidence), self.settings.bedrock_model_id
        return super()._synthesize(request, evidence)


def ensure_current_sources(item, request, principal):
    if item["kind"] == "entity":
        links = item["links"]
    elif item["kind"] == "structured-record":
        content = item["content"]
        if any(s["grounding_status"] != "verified" for s in content["statements"]):
            raise HTTPException(409, "Uncited statements must be resolved before approval/export.")
        links = [e for e in content["evidence"] if e["evidence_id"].startswith("workspace:")]
    elif item["kind"] == "review":
        links = [
            e
            for e in item["analysis"]["report"]["evidence"]
            if e["evidence_id"].startswith("workspace:")
        ]
    else:
        return
    for link in links:
        doc = document_record(request, principal, link["document_id"])
        if doc["version"] != link["version"]:
            raise HTTPException(409, "Supporting source changed. Create a fresh draft.")


class WorkspaceRetrieval:
    def __init__(self, db, principal, settings, document_id=None, references_only=False):
        self.db, self.principal, self.settings = db, principal, settings
        self.document_id = document_id
        self.references_only = references_only

    def manifest(self):
        return FederalCorpusRetrieval(self.settings.local_corpus_path).manifest()

    def relations(self, document_ids):
        return FederalCorpusRetrieval(self.settings.local_corpus_path).relations(document_ids)

    def retrieve(self, queries, principal=None, filters=None):
        query = " ".join(queries)
        terms = set(re.findall(r"\w+", query.casefold()))
        ranked = []
        for doc in self.db.list(self.principal, "document"):
            if doc.get("deleted") or doc.get("status") == "source-error":
                continue
            if self.references_only and doc["category"] == "contract":
                continue
            if self.document_id and doc["id"] != self.document_id and doc["category"] == "contract":
                continue
            for index, text in enumerate(re.findall(r"[\s\S]{1,1500}", doc["text"])):
                score = len(terms & set(re.findall(r"\w+", text.casefold())))
                if doc["id"] == self.document_id:
                    score += 2
                if score:
                    ranked.append(
                        (
                            score,
                            Evidence(
                                evidence_id=f"workspace:{doc['id']}:{doc['version']}:{index}",
                                document_id=doc["id"],
                                source=doc["category"],
                                title=doc["title"],
                                excerpt=text,
                                version=doc["version"],
                                url=doc.get("source_url") or None,
                                acl_principals=[f"user:{self.principal.subject}"],
                                security_label=self.principal.security_domain,
                            ),
                        )
                    )
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        local = [item for _, item in ranked[:4]]
        federal = []
        if self.settings.local_corpus_path:
            federal = FederalCorpusRetrieval(self.settings.local_corpus_path, top_k=4).retrieve(
                queries
            )
        return local + federal


@router.post("/questions")
async def ask(
    payload: QuestionInput, request: Request, principal: Principal, settings: Configuration
):
    query = payload.query
    if payload.document_id:
        doc = document_record(request, principal, payload.document_id)
        query = f"Contract: {doc['title']}. Request: {query}"
    if payload.finding_id:
        matches = [
            f
            for review in store(request).list(principal, "review")
            if review["document_id"] == payload.document_id
            for f in review["analysis"]["findings"]
            if f["id"] == payload.finding_id
        ]
        if not matches:
            raise HTTPException(404, "Finding not available in this contract")
        query += "\nFinding to investigate (not an authority): " + matches[0]["title"]
    retrieval = WorkspaceRetrieval(store(request), principal, settings, payload.document_id)
    # No synthetic fallback: only authorized workspace text and official corpus.
    service = WorkspaceGeneration(settings, retrieval=retrieval)
    async with request.app.state.review_semaphore:
        try:
            response = await asyncio.to_thread(
                service.generate,
                IntelligenceQuery(
                    query=query[:4000], mode=payload.mode, workflow="contract-review"
                ),
                principal,
            )
        except Exception as error:
            raise HTTPException(
                503, "Cited generation unavailable. Check corpus and model access."
            ) from error
    return store(request).save(
        principal,
        "question",
        {
            "query": payload.query,
            "document_id": payload.document_id,
            "finding_id": payload.finding_id,
            "response": response.model_dump(mode="json"),
        },
    )


@router.get("/questions")
def questions(request: Request, principal: Principal):
    return store(request).list(principal, "question")


@router.get("/events")
def events(request: Request, principal: Principal):
    return store(request).list(principal, "event")[:200]


class ConnectionInput(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    provider: Literal["shared-folder", "sharepoint"]
    folder: str = Field(default="", max_length=500)
    category: Literal["contract", "policy", "playbook", "clause-library", "reference"] = "reference"


@router.get("/connections")
def connections(request: Request, principal: Principal, settings: Configuration):
    return {
        "connections": store(request).list(principal, "connection"),
        "shared_folder_available": bool(settings.workspace_import_root),
        "sharepoint_available": bool(settings.graph_connector_secret_arn),
        "automatic_sync_seconds": 0,
        "persistence": "DynamoDB" if settings.workspace_table else "Local SQLite",
        "security_domain": principal.security_domain,
    }


def allowed_folder(settings, relative):
    if not settings.workspace_import_root:
        raise HTTPException(409, "Administrator must configure WORKSPACE_IMPORT_ROOT first.")
    root = Path(settings.workspace_import_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise HTTPException(422, "Choose a folder within the approved import root.")
    return path


@router.post("/connections")
def connect(
    payload: ConnectionInput, request: Request, principal: Principal, settings: Configuration
):
    if payload.provider == "shared-folder":
        allowed_folder(settings, payload.folder)
    elif not settings.graph_connector_secret_arn:
        raise HTTPException(409, "SharePoint requires an administrator-configured Graph secret.")
    return store(request).save(
        principal,
        "connection",
        {**payload.model_dump(), "status": "idle", "last_sync": None, "errors": [], "counts": {}},
    )


def source_files(connection, settings, principal):
    if connection["provider"] == "shared-folder":
        root = allowed_folder(settings, connection["folder"])
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.is_symlink()
                or not path.resolve().is_relative_to(root)
                or path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}
            ):
                continue
            key = str(path.relative_to(root))
            if path.stat().st_size > settings.max_upload_bytes:
                yield key, None, "File exceeds upload size limit"
                continue
            try:
                data = path.read_bytes()
                doc = (
                    DocumentParser(settings).parse(path.name, data)
                    if path.suffix.lower() in {".pdf", ".docx"}
                    else parsed_text(path.name, data.decode("utf-8"))
                )
                yield key, doc, None
            except (ValueError, UnicodeError, OSError):
                yield key, None, "Document could not be parsed"
    else:
        import boto3

        from ingestion.graph import GraphClient, GraphDeltaConnector

        value = boto3.client("secretsmanager", region_name=settings.aws_region).get_secret_value(
            SecretId=settings.graph_connector_secret_arn
        )["SecretString"]
        config = json.loads(value)
        client = GraphClient(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            tenant_id=config["tenant_id"],
            graph_base_url=config.get("graph_base_url", "https://graph.microsoft.us/v1.0"),
            token_url=config.get("token_url"),
        )
        connector = GraphDeltaConnector(
            client, drive_id=config["drive_id"], corpus="lens", default_acl_principals=()
        )
        # Full listing on each sync makes delete reconciliation reliable across interrupted jobs.
        batch = connector.changes()
        for event in batch.events:
            doc = event.document
            if doc is None or not (set(doc.acl_principals) & principal.acl_principals):
                continue
            if doc.security_label not in {"public", principal.security_domain}:
                continue
            yield (
                doc.document_id,
                parsed_text(doc.provenance.get("name", doc.document_id), doc.text),
                None,
            )


def sync_connection(db, principal, connection_id, settings):
    connection = db.get(principal, connection_id)
    counts = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0}
    errors = []
    try:
        previous = {
            d.get("source_key"): d
            for d in db.list(principal, "document")
            if d.get("connection_id") == connection_id
        }
        seen = set()
        for key, document, error in source_files(connection, settings, principal):
            seen.add(key)
            if error:
                errors.append({"file": key, "message": error})
                if old := previous.get(key):
                    old["status"] = "source-error"
                    db.save(principal, "document", old, old["id"], old["revision"])
                continue
            old = previous.get(key)
            if (
                old
                and old["sha256"] == document.sha256
                and not old.get("deleted")
                and old.get("status") != "source-error"
            ):
                counts["unchanged"] += 1
                continue
            try:
                save_document(
                    db,
                    principal,
                    document,
                    connection["category"],
                    old.get("metadata") if old else None,
                    record_id=old["id"] if old else None,
                    connection_id=connection_id,
                    source_key=key,
                )
                counts["updated" if old else "added"] += 1
            except HTTPException as error:
                errors.append({"file": key, "message": str(error.detail)})
        for key, old in previous.items():
            if key not in seen and not old.get("deleted"):
                old["deleted"] = True
                db.save(principal, "document", old, old["id"], old["revision"])
                db.event(principal, "document.removed_from_source", old["id"], {})
                counts["removed"] += 1
        connection.update(
            status="complete" if not errors else "partial",
            counts=counts,
            errors=errors[:100],
            last_sync=now(),
        )
    except Exception:
        connection.update(
            status="failed",
            errors=[
                {
                    "file": "connection",
                    "message": "Sync failed. Check source access and connector configuration.",
                }
            ],
            last_sync=now(),
        )
    db.save(principal, "connection", connection, connection_id)
    db.event(
        principal,
        "connection.synced",
        connection_id,
        {"counts": counts, "status": connection["status"]},
    )


@router.post("/connections/{connection_id}/sync")
def sync(
    connection_id: str,
    request: Request,
    principal: Principal,
    settings: Configuration,
    tasks: BackgroundTasks,
):
    item = store(request).get(principal, connection_id)
    if item["kind"] != "connection":
        raise HTTPException(404, "Connection not found")
    if item["status"] == "syncing":
        raise HTTPException(409, "A sync is already running")
    item["status"] = "syncing"
    updated = store(request).save(principal, "connection", item, connection_id, item["revision"])
    tasks.add_task(sync_connection, store(request), principal, connection_id, settings)
    return updated


@router.get("/library")
def library(request: Request, principal: Principal, settings: Configuration, q: str = ""):
    corpus = None
    evidence = []
    if settings.local_corpus_path:
        try:
            retrieval = FederalCorpusRetrieval(settings.local_corpus_path)
            corpus = retrieval.manifest()
            if q.strip():
                evidence = [e.model_dump(mode="json") for e in retrieval.retrieve([q[:1000]])]
        except (RuntimeError, OSError):
            corpus = None
    catalog_path = Path(__file__).resolve().parents[1] / "knowledge/source_catalog.json"
    return {
        "corpus": corpus,
        "evidence": evidence,
        "catalog": json.loads(catalog_path.read_text()),
        "documents": documents(request, principal),
    }


class EntityInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    entity_type: Literal["vendor", "agency", "organization", "obligation", "contract"]
    document_id: str
    excerpt: str = Field(min_length=3, max_length=1500)
    relationship: str = Field(default="mentioned in", max_length=100)


@router.get("/entities")
def entities(request: Request, principal: Principal):
    return store(request).list(principal, "entity")


@router.post("/entities")
def entity(payload: EntityInput, request: Request, principal: Principal):
    doc = document_record(request, principal, payload.document_id)
    if payload.excerpt not in doc["text"]:
        raise HTTPException(422, "The supporting excerpt must occur in the selected document.")
    key = hashlib.sha256(
        f"{payload.entity_type}:{payload.name.strip().casefold()}".encode()
    ).hexdigest()
    existing = [e for e in store(request).list(principal, "entity") if e.get("match_key") == key]
    previous = existing[0] if existing else None
    links = previous["links"] if previous else []
    link = {
        "document_id": doc["id"],
        "version": doc["version"],
        "excerpt": payload.excerpt,
        "relationship": payload.relationship,
    }
    if link not in links:
        links.append(link)
    item = store(request).save(
        principal,
        "entity",
        {
            "name": payload.name,
            "entity_type": payload.entity_type,
            "match_key": key,
            "links": links,
            "decision": "draft",
            "note": "",
            "method": "analyst-submitted; exact normalized-name match",
        },
        previous["id"] if previous else None,
        previous["revision"] if previous else 0,
    )
    store(request).event(
        principal, "entity.updated" if previous else "entity.created", item["id"], {}
    )
    return item


class RecordInput(BaseModel):
    question_id: str
    title: str = Field(min_length=1, max_length=200)


@router.get("/structured-records")
def records(request: Request, principal: Principal):
    return store(request).list(principal, "structured-record")


@router.post("/structured-records")
def create_record(payload: RecordInput, request: Request, principal: Principal):
    question = store(request).get(principal, payload.question_id)
    if question["kind"] != "question" or not question["response"]["statements"]:
        raise HTTPException(422, "Generate a cited answer or draft before creating a record.")
    return store(request).save(
        principal,
        "structured-record",
        {
            "title": payload.title,
            "question_id": question["id"],
            "document_id": question.get("document_id"),
            "content": question["response"],
            "decision": "draft",
            "note": "",
        },
    )
