"""FastAPI routes for programmatic contract review."""

import asyncio
import json
import secrets
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)

from app.config import Settings, get_settings
from app.identity import decode_principal, issue_demo_token
from app.intelligence import (
    CitedGenerationService,
    DynamoDBIntelligenceRepository,
    IntelligenceRepository,
    ingestion_operational_status,
)
from app.models import (
    AcquisitionMetadata,
    AnalystDecisionRequest,
    EntityResolutionRequest,
    IntelligenceEntity,
    IntelligenceQuery,
    IntelligenceResponse,
    PrincipalContext,
    ReviewReport,
)
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata
from app.service import ReviewService

router = APIRouter()
basic_auth = HTTPBasic(auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)
intelligence_repository = IntelligenceRepository()
_aws_repository: DynamoDBIntelligenceRepository | None = None


def require_judge_credentials(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(basic_auth)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    """Authenticate API callers without timing-sensitive string comparison."""

    valid_username = credentials is not None and secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.workspace_username.encode("utf-8"),
    )
    valid_password = credentials is not None and secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.workspace_password.get_secret_value().encode("utf-8"),
    )
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def get_review_service(
    request: Request,
) -> ReviewService:
    return request.app.state.review_service


def require_intelligence_principal(
    bearer: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_auth),
    ],
    basic: Annotated[HTTPBasicCredentials | None, Depends(basic_auth)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PrincipalContext:
    if bearer is not None:
        return decode_principal(bearer.credentials, settings)
    username = require_judge_credentials(basic, settings)
    return PrincipalContext(
        subject=username,
        groups=["contract-reviewers", "mission-analysts"],
        security_domain=settings.security_domain,
        scopes=[
            "rag:query",
            "entities:read",
        ],
    )


def get_cited_generation_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> CitedGenerationService:
    return CitedGenerationService(settings)


def get_intelligence_repository(
    settings: Annotated[Settings, Depends(get_settings)],
) -> IntelligenceRepository:
    global _aws_repository
    table_names = (
        settings.entity_registry_table,
        settings.change_event_table,
    )
    if not all(table_names):
        return intelligence_repository
    if _aws_repository is None:
        _aws_repository = DynamoDBIntelligenceRepository(
            entity_table=settings.entity_registry_table,
            change_table=settings.change_event_table,
            region=settings.aws_region,
        )
    return _aws_repository


@router.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/v1/auth/demo-token", tags=["identity"])
def create_demo_token(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, str | int]:
    return {
        "access_token": issue_demo_token(username, password, settings),
        "token_type": "bearer",
        "expires_in": 1800,
    }


@router.post(
    "/api/v1/intelligence/query",
    response_model=IntelligenceResponse,
    tags=["intelligence"],
)
async def query_intelligence(
    payload: IntelligenceQuery,
    service: Annotated[
        CitedGenerationService,
        Depends(get_cited_generation_service),
    ],
    repository: Annotated[
        IntelligenceRepository,
        Depends(get_intelligence_repository),
    ],
    principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> IntelligenceResponse:
    if "rag:query" not in principal.scopes:
        raise HTTPException(status_code=403, detail="rag:query scope is required")
    response = await asyncio.to_thread(service.generate, payload, principal)
    repository.register_evidence(response.evidence)
    return response


@router.get("/api/v1/intelligence/ingestion/status", tags=["intelligence"])
def ingestion_status(
    settings: Annotated[Settings, Depends(get_settings)],
    _principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> dict[str, object]:
    return ingestion_operational_status(settings)


@router.post(
    "/api/v1/intelligence/entities/resolve",
    response_model=list[IntelligenceEntity],
    tags=["intelligence"],
)
def resolve_entities(
    payload: EntityResolutionRequest,
    repository: Annotated[
        IntelligenceRepository,
        Depends(get_intelligence_repository),
    ],
    principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> list[IntelligenceEntity]:
    if "entities:read" not in principal.scopes:
        raise HTTPException(status_code=403, detail="entities:read scope is required")
    return repository.resolve(payload.candidates)


@router.get(
    "/api/v1/intelligence/entities",
    response_model=list[IntelligenceEntity],
    tags=["intelligence"],
)
def list_entities(
    repository: Annotated[
        IntelligenceRepository,
        Depends(get_intelligence_repository),
    ],
    principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> list[IntelligenceEntity]:
    if "entities:read" not in principal.scopes:
        raise HTTPException(status_code=403, detail="entities:read scope is required")
    return repository.list_entities()


@router.get("/api/v1/intelligence/changes", tags=["intelligence"])
def list_changes(
    repository: Annotated[
        IntelligenceRepository,
        Depends(get_intelligence_repository),
    ],
    principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> list[dict[str, object]]:
    if "entities:read" not in principal.scopes:
        raise HTTPException(status_code=403, detail="entities:read scope is required")
    return [item.model_dump(mode="json") for item in repository.list_changes()]


@router.get("/api/v1/intelligence/relationships", tags=["intelligence"])
def list_relationships(
    repository: Annotated[
        IntelligenceRepository,
        Depends(get_intelligence_repository),
    ],
    principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> list[dict[str, object]]:
    if "entities:read" not in principal.scopes:
        raise HTTPException(status_code=403, detail="entities:read scope is required")
    return [item.model_dump(mode="json") for item in repository.list_relationships()]


@router.post(
    "/api/v1/intelligence/entities/{entity_id}/decision",
    response_model=IntelligenceEntity,
    tags=["intelligence"],
)
def decide_entity(
    entity_id: str,
    payload: AnalystDecisionRequest,
    repository: Annotated[
        IntelligenceRepository,
        Depends(get_intelligence_repository),
    ],
    principal: Annotated[
        PrincipalContext,
        Depends(require_intelligence_principal),
    ],
) -> IntelligenceEntity:
    try:
        return repository.decide_entity(entity_id, payload, principal)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/v1/reviews", response_model=ReviewReport, tags=["reviews"])
async def review_contract(
    request: Request,
    file: Annotated[UploadFile, File()],
    metadata_json: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[ReviewService, Depends(get_review_service)],
    _authenticated_user: Annotated[str, Depends(require_judge_credentials)],
) -> ReviewReport:
    try:
        metadata = AcquisitionMetadata.model_validate(json.loads(metadata_json))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid acquisition metadata") from exc
    parser = DocumentParser(settings)
    data = await parser.read_upload(file)
    try:
        async with request.app.state.review_semaphore:
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
    return report


@router.post("/api/v1/reviews/sample", response_model=ReviewReport, tags=["reviews"])
async def review_sample(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[ReviewService, Depends(get_review_service)],
    _authenticated_user: Annotated[str, Depends(require_judge_credentials)],
) -> ReviewReport:
    async with request.app.state.review_semaphore:
        report = await asyncio.to_thread(
            _parse_and_review,
            DocumentParser(settings),
            service,
            "sample-contract.docx",
            sample_contract_bytes(),
            sample_metadata(),
        )
    return report


def _parse_and_review(
    parser: DocumentParser,
    service: ReviewService,
    filename: str,
    data: bytes,
    metadata: AcquisitionMetadata,
) -> ReviewReport:
    document = parser.parse_isolated(filename, data)
    return service.review(document, metadata)
