"""ASGI entrypoint for the Acquisition Lens and its APIs."""

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import Settings, get_settings
from app.lens import router as lens_router
from app.service import ReviewService
from app.workspace import router as workspace_router
from app.workspace_store import WorkspaceStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    for variable in ("HOME",):
        configured_path = os.getenv(variable)
        if configured_path:
            Path(configured_path).mkdir(parents=True, exist_ok=True)
    api = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description=("Contract-first, source-grounded review and J2 intelligence workflow demo."),
    )
    api.state.review_semaphore = asyncio.Semaphore(settings.max_review_concurrency)
    api.state.workspace_store = WorkspaceStore(settings)
    if settings.model_review_enabled:
        from app.model_review import ModelReviewService

        api.state.review_service = ModelReviewService(settings)
    else:
        api.state.review_service = ReviewService(settings)
    api.dependency_overrides[get_settings] = lambda: settings
    api.include_router(router)
    api.include_router(lens_router)
    api.include_router(workspace_router)
    lens_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if lens_dist.is_dir():
        api.mount("/lens", StaticFiles(directory=lens_dist, html=True), name="lens")

    @api.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/lens/")

    return api


app = create_app()
