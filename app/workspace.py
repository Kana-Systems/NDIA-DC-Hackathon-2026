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
from pydantic import BaseModel, Field, field_validator

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
from app.workspace_export import WorkspaceExport, build_export
from app.workspace_readiness import ReadinessEvaluator
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
    # ParsedDocument joins segments with blank lines. Split only at those existing
    # boundaries so TXT sources are reconstructed verbatim, never mid-word.
    paragraphs = text.split("\n\n")
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

    @field_validator("title", "text")
    @classmethod
    def meaningful_input(cls, value, info):
        minimum = 80 if info.field_name == "text" else 1
        if len(value.strip()) < minimum:
            raise ValueError(f"Add at least {minimum} non-padding characters")
        return value  # Preserve the exact supplied text for excerpts and hashes.


def save_document(db, principal, document, category, metadata=None, **extra):
    limit = db.settings.max_extracted_characters if db.s3 is not None else 30000
    if len(document.text) > limit:
        raise HTTPException(413, f"Source exceeds this deployment's {limit:,}-character limit.")
    record_id = extra.pop("record_id", None)
    raw_bytes = extra.pop("raw_bytes", None)
    payload = {
        "title": document.filename,
        "text": document.text,
        "sha256": document.sha256,
        "parsed": document.model_dump(mode="json"),
        "category": category,
        "metadata": metadata,
        "status": "indexing" if db.search else "ready",
        "deleted": False,
        "version": document.sha256[:12],
        **extra,
    }
    item = db.save(principal, "document", payload, record_id)
    if raw_bytes is not None and db.s3 is not None:
        item["original_key"] = db.put_blob(principal, item["id"], raw_bytes, document.media_type)
        item = db.save(principal, "document", item, item["id"], item["revision"])
    if db.search:
        item = index_document(db, principal, item)
    action = "document.index_failed" if item["status"] == "index-failed" else "document.indexed"
    db.event(principal, action, item["id"], {"version": item["version"]})
    return item


def index_document(db, principal, item):
    try:
        db.search.index_document(item)
        item["status"] = "ready"
        item.pop("index_error", None)
    except Exception:
        item["status"] = "index-failed"
        item["index_error"] = "Saved source; search indexing failed. Retry indexing."
    return db.save(principal, "document", item, item["id"], item["revision"])


@router.post("/documents/{record_id}/reindex")
def reindex(record_id: str, request: Request, principal: Principal):
    item = document_record(request, principal, record_id)
    db = store(request)
    if not db.search:
        raise HTTPException(409, "OpenSearch is not configured")
    return index_document(db, principal, item)


@router.get("/documents")
def documents(request: Request, principal: Principal):
    db = store(request)
    return [
        {
            **{k: v for k, v in item.items() if k not in {"text", "parsed"}},
            "available": db.source_available(principal, item),
        }
        for item in db.list(principal, "document")
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
    return await asyncio.to_thread(
        save_document, store(request), principal, document, category, raw_bytes=data
    )


@router.get("/documents/{record_id}")
def get_document(record_id: str, request: Request, principal: Principal):
    item = document_record(request, principal, record_id)
    return {
        **{k: v for k, v in item.items() if k != "parsed"},
        "available": store(request).source_available(principal, item),
    }


class MetadataInput(BaseModel):
    metadata: AcquisitionMetadata
    revision: int


@router.put("/documents/{record_id}/metadata")
def metadata(record_id: str, payload: MetadataInput, request: Request, principal: Principal):
    item = document_record(request, principal, record_id)
    item["metadata"] = payload.metadata.model_dump(mode="json")
    return store(request).save(principal, "document", item, record_id, payload.revision)


@router.post("/documents/{record_id}/review")
async def review(record_id: str, request: Request, principal: Principal, settings: Configuration):
    item = document_record(request, principal, record_id)
    if not store(request).source_available(principal, item):
        raise HTTPException(409, "Sync or index this source successfully before review")
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
    evaluator = ReadinessEvaluator(store(request), principal)
    return [
        evaluator.present(item)
        for item in sorted(
            store(request).list(principal, "review"), key=lambda r: r["created_at"], reverse=True
        )
    ]


class DecisionInput(BaseModel):
    decision: Literal["approved", "rejected"]
    note: str = Field(min_length=3, max_length=2000)
    revision: int

    @field_validator("note")
    @classmethod
    def meaningful_note(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Record what you checked in the decision note")
        return value


@router.post("/records/{record_id}/decision")
def decide(record_id: str, payload: DecisionInput, request: Request, principal: Principal):
    item = store(request).get(principal, record_id)
    if item["kind"] not in {"review", "entity", "structured-record"}:
        raise HTTPException(422, "This record cannot be approved")
    if payload.decision == "approved":
        ensure_current_sources(item, request, principal)
    item.update(
        decision=payload.decision, note=payload.note, decided_by=principal.subject, decided_at=now()
    )
    result = store(request).save(principal, item["kind"], item, record_id, payload.revision)
    store(request).event(principal, "analyst." + payload.decision, record_id, {})
    return result


@router.get("/export-schema")
def export_schema(principal: Principal):
    return WorkspaceExport.model_json_schema()


@router.get("/records/{record_id}/export", response_model=WorkspaceExport)
def export(record_id: str, request: Request, principal: Principal):
    item = store(request).get(principal, record_id)
    if item["kind"] not in {"review", "structured-record"} or item.get("decision") != "approved":
        raise HTTPException(409, "Approve the record before exporting.")
    result = build_export(item, ReadinessEvaluator(store(request), principal))
    store(request).event(principal, "record.exported", record_id, {})
    return result


class QuestionInput(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    mode: Literal["answer", "summary", "draft"] = "answer"
    document_id: str | None = None
    finding_id: str | None = None

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("Enter a question or drafting request")
        return value


class WorkspaceGeneration(CitedGenerationService):
    """Live LLM errors are visible; explicit offline mode quotes actual evidence."""

    def _synthesize(self, request, evidence):
        if self.settings.bedrock_enabled:
            return self._remote_synthesis(request, evidence), self.settings.bedrock_model_id
        return super()._synthesize(request, evidence)


def ensure_current_sources(item, request, principal):
    ReadinessEvaluator(store(request), principal).require_current(item)


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
        if self.db.search:
            local = self.db.search.retrieve(
                queries, self.principal, self.db, self.document_id, self.references_only
            )
            federal = self.federal(queries)
            return local + federal
        query = " ".join(queries)
        terms = set(re.findall(r"\w+", query.casefold()))
        ranked = []
        for doc in self.db.list(self.principal, "document"):
            if not self.db.source_available(self.principal, doc):
                continue
            if self.references_only and doc["category"] == "contract":
                continue
            if self.document_id and doc["id"] != self.document_id and doc["category"] == "contract":
                continue
            if "text" not in doc:
                doc = self.db.get(self.principal, doc["id"])
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
        return local + self.federal(queries)

    def federal(self, queries):
        if self.db.search and self.settings.local_corpus_path:
            return self.db.search.retrieve_federal(queries)
        federal = []
        if self.settings.local_corpus_path:
            federal = FederalCorpusRetrieval(self.settings.local_corpus_path, top_k=4).retrieve(
                queries
            )
        return federal


@router.post("/questions")
async def ask(
    payload: QuestionInput, request: Request, principal: Principal, settings: Configuration
):
    query = payload.query
    document_version = None
    workflow = "mission-support"
    if payload.document_id:
        doc = document_record(request, principal, payload.document_id)
        if not store(request).source_available(principal, doc):
            raise HTTPException(
                409, "Sync or index this source successfully before asking about it"
            )
        query = f"Source document: {doc['title']}. Request: {query}"
        document_version = doc["version"]
        if doc["category"] == "contract":
            workflow = "contract-review"
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
                IntelligenceQuery(query=query[:4000], mode=payload.mode, workflow=workflow),
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
            "document_version": document_version,
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
        "automatic_sync_seconds": settings.workspace_sync_seconds,
        "search": "OpenSearch hybrid" if settings.workspace_search_enabled else "Local lexical",
        "document_storage": "S3" if settings.workspace_source_bucket else "Local SQLite",
        "persistence": "DynamoDB" if settings.workspace_table else "Local SQLite",
        "security_domain": principal.security_domain,
    }


@router.post("/connections/sharepoint/check")
def check_sharepoint(principal: Principal, settings: Configuration):
    from app.sharepoint import check_connection

    try:
        return check_connection(settings, principal)[2]
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            503, "SharePoint check failed. Verify credentials and selected-site read grant."
        ) from error


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
    else:
        check_sharepoint(principal, settings)
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
                yield key, None, "File exceeds upload size limit", {}
                continue
            try:
                data = path.read_bytes()
                doc = (
                    DocumentParser(settings).parse(path.name, data)
                    if path.suffix.lower() in {".pdf", ".docx"}
                    else parsed_text(path.name, data.decode("utf-8"))
                )
                yield key, doc, None, {"raw_bytes": data}
            except (ValueError, UnicodeError, OSError):
                yield key, None, "Document could not be parsed", {}
    else:
        from app.sharepoint import source_files as sharepoint_files

        yield from sharepoint_files(connection, settings, principal)


def sync_connection(db, principal, connection_id, settings, lease=None):
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
        for key, document, error, provenance in source_files(connection, settings, principal):
            if lease and db.get(principal, connection_id)["sync_lease"] != lease:
                return
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
                and old.get("status") == "ready"
                and old.get("source_version") == provenance.get("source_version")
            ):
                counts["unchanged"] += 1
                continue
            try:
                saved = save_document(
                    db,
                    principal,
                    document,
                    connection["category"],
                    old.get("metadata") if old else None,
                    record_id=old["id"] if old else None,
                    connection_id=connection_id,
                    source_key=key,
                    **provenance,
                )
                if saved["status"] == "index-failed":
                    errors.append({"file": key, "message": saved["index_error"]})
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
    latest = db.get(principal, connection_id)
    if lease and latest.get("sync_lease") != lease:
        return
    db.save(principal, "connection", connection, connection_id, latest["revision"])
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
    from app.workspace_sync import claim

    updated = claim(store(request), principal, item)
    tasks.add_task(
        sync_connection, store(request), principal, connection_id, settings, updated["sync_lease"]
    )
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
                found = (
                    store(request).search.retrieve_federal([q[:1000]])
                    if store(request).search
                    else retrieval.retrieve([q[:1000]])
                )
                evidence = [e.model_dump(mode="json") for e in found]
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
    evaluator = ReadinessEvaluator(store(request), principal)
    return [evaluator.present(item) for item in store(request).list(principal, "entity")]


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
    evaluator = ReadinessEvaluator(store(request), principal)
    return [evaluator.present(item) for item in store(request).list(principal, "structured-record")]


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
            "document_version": question.get("document_version"),
            "content": question["response"],
            "decision": "draft",
            "note": "",
        },
    )
