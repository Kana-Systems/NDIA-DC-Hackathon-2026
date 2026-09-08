"""ASGI entrypoint that mounts the API and authenticated Gradio UI."""

import asyncio
import os
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import Settings, get_settings
from app.lens import router as lens_router
from app.ui import build_ui


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    for variable in ("HOME", "GRADIO_TEMP_DIR"):
        configured_path = os.getenv(variable)
        if configured_path:
            Path(configured_path).mkdir(parents=True, exist_ok=True)
    api = FastAPI(
        title=settings.app_name,
        version="0.2.0",
        description=("Contract-first, source-grounded review and J2 intelligence workflow demo."),
    )
    api.state.review_semaphore = asyncio.Semaphore(settings.max_review_concurrency)
    api.dependency_overrides[get_settings] = lambda: settings
    api.include_router(router)
    api.include_router(lens_router)
    lens_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if lens_dist.is_dir():
        api.mount("/lens", StaticFiles(directory=lens_dist, html=True), name="lens")

    @api.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/ui/")

    return gr.mount_gradio_app(
        api,
        build_ui(settings),
        path="/ui",
        auth=(
            settings.gradio_username,
            settings.gradio_password.get_secret_value(),
        ),
        max_file_size=f"{settings.max_upload_mb}mb",
        show_error=False,
    )


app = create_app()
