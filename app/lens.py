"""Acquisition Lens browser adapter over the canonical review service."""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from app.api import _parse_and_review, get_review_service, require_intelligence_principal
from app.config import Settings, get_settings
from app.identity import issue_demo_token
from app.models import (
    AcquisitionMetadata,
    DocumentLocation,
    ExtractedSegment,
    ParsedDocument,
    PrincipalContext,
    ReviewReport,
)
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
from app.service import ReviewService

router = APIRouter(tags=["acquisition-lens"])


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


class TextReviewRequest(BaseModel):
    title: str = Field(default="Contract review", min_length=1, max_length=200)
    text: str = Field(min_length=80, max_length=200_000)
    metadata: AcquisitionMetadata

    @field_validator("text")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if len(value.strip()) < 80:
            raise ValueError("Add at least 80 characters of contract text")
        return value


def require_reviewer(
    principal: Annotated[PrincipalContext, Depends(require_intelligence_principal)],
) -> PrincipalContext:
    if "contract-reviewers" not in principal.groups:
        raise HTTPException(status_code=403, detail="Contract reviewer access is required")
    return principal


@router.post("/api/auth/login")
def login(
    payload: LoginRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str | int]:
    return {
        "access_token": issue_demo_token(settings.gradio_username, payload.password, settings),
        "token_type": "bearer",
        "expires_in": 1800,
    }


@router.get("/api/demo/sample", dependencies=[Depends(require_reviewer)])
def sample(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, object]:
    document = DocumentParser(settings).parse("sample.docx", sample_contract_bytes())
    return {
        "title": "Sample defense services contract",
        "text": document.text,
        "metadata": sample_metadata().model_dump(mode="json"),
    }


@router.get("/api/sources", dependencies=[Depends(require_reviewer)])
def sources() -> list[dict[str, object]]:
    path = Path(__file__).resolve().parent.parent / "knowledge" / "source_catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def present_report(report: ReviewReport) -> dict[str, object]:
    """Map display fields without inventing confidence or strengthening citations."""
    evidence = {item.evidence_id: item for item in report.evidence}
    priorities = ["critical", "high", "medium", "low", "info"]
    return {
        "document_summary": report.executive_summary,
        "overall_risk": next(
            (
                severity
                for severity in priorities
                if any(f.severity.value == severity for f in report.findings)
            ),
            "info",
        ),
        "confidence": None,
        "engine": report.synthesis_mode,
        "disclaimer": (
            "Screening only. Retrieved citations do not establish applicability. "
            "Classifier scores are not probabilities of legal correctness. Human review required."
        ),
        "findings": [
            {
                "id": finding.finding_id,
                "category": finding.rule_id,
                "title": finding.title,
                "severity": finding.severity.value,
                "confidence": finding.classifier_confidence,
                "excerpt": "\n\n".join(c.text for c in finding.document_context),
                "explanation": finding.description,
                "recommendation": finding.recommendation,
                "grounding_status": finding.grounding_status.value,
                "classifier_model_ids": finding.classifier_model_ids,
                "citations": [
                    {
                        "title": evidence[citation].title,
                        "url": evidence[citation].url,
                        "section": evidence[citation].source,
                        "excerpt": evidence[citation].excerpt,
                        "verification_status": "retrieved_evidence",
                    }
                    for citation in finding.citation_ids
                    if citation in evidence
                ],
            }
            for finding in report.findings
        ],
        "report": report.model_dump(mode="json"),
    }


@router.post("/api/analyze", dependencies=[Depends(require_reviewer)])
async def analyze(
    payload: TextReviewRequest,
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> dict[str, object]:
    if len(payload.text) > settings.max_extracted_characters:
        raise HTTPException(status_code=413, detail="Document exceeds the text limit")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n|\n", payload.text) if part.strip()]
    if len(paragraphs) > settings.max_document_segments:
        raise HTTPException(status_code=413, detail="Document exceeds the segment limit")
    document = ParsedDocument(
        filename=payload.title,
        media_type="text/plain",
        sha256=hashlib.sha256(payload.text.encode()).hexdigest(),
        segments=[
            ExtractedSegment(
                segment_id=f"text-{index}",
                text=paragraph,
                location=DocumentLocation(paragraph=index, label=f"Paragraph {index}"),
            )
            for index, paragraph in enumerate(paragraphs, 1)
        ],
    )
    async with request.app.state.review_semaphore:
        try:
            report = await asyncio.to_thread(service.review, document, payload.metadata)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model review failed. Verify Bedrock access, model artifacts, "
                    "and the federal corpus."
                ),
            ) from exc
    return present_report(report)


@router.post("/api/review-upload", dependencies=[Depends(require_reviewer)])
async def review_upload(
    request: Request,
    file: Annotated[UploadFile, File()],
    metadata_json: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[ReviewService, Depends(get_review_service)],
) -> dict[str, object]:
    try:
        metadata = AcquisitionMetadata.model_validate_json(metadata_json)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid acquisition metadata") from exc
    parser = DocumentParser(settings)
    data = await parser.read_upload(file)
    async with request.app.state.review_semaphore:
        try:
            report = await asyncio.to_thread(
                _parse_and_review,
                parser,
                service,
                file.filename or "upload",
                data,
                metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model review is unavailable") from exc
    return present_report(report)
