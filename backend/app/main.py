"""
FastAPI application factory.

Initialises the application, sets up CORS, mounts static files
and Jinja2 templates, registers API routers, and initialises
the database on startup.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.app.utils.config import get_settings
from backend.app.utils.logging import get_logger, setup_logging

# ── Logging must be configured first ─────────────────────────────────────────
settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

# ── Template / static directories ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting Sentinel-2 Change Detection System...")

    # Initialise database
    from backend.app.storage.database import init_database
    init_database()
    logger.info("Database ready.")

    # Check credentials
    if settings.has_copernicus_credentials:
        logger.info("Copernicus credentials: configured ✓")
    else:
        logger.warning(
            "Copernicus credentials NOT configured. "
            "Satellite imagery download will fail. "
            "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in .env"
        )

    yield

    logger.info("Sentinel-2 Change Detection System shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Sentinel-2 Change Detection System",
        description=(
            "Local-first satellite imagery change detection using "
            "Copernicus Sentinel-2 data. Compares multi-temporal imagery, "
            "computes NDVI/NDWI/NDBI, and detects significant spectral changes."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files ──────────────────────────────────────────────────────────
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Mount data directory for serving result images
    data_dir = settings.data_dir
    if data_dir.exists():
        app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

    # ── API routers ───────────────────────────────────────────────────────────
    from backend.app.api import imagery, changes, indices as indices_api, models_api

    app.include_router(imagery.router, prefix="/api", tags=["Imagery"])
    app.include_router(changes.router, prefix="/api", tags=["Analysis"])
    app.include_router(indices_api.router, prefix="/api", tags=["Indices"])
    app.include_router(models_api.router, prefix="/api", tags=["Models"])

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get("/api/health", tags=["System"])
    async def health():
        """Health check endpoint."""
        gpu_available = False
        gpu_name = None
        gpu_vram = None
        pytorch_available = False
        try:
            import torch
            pytorch_available = True
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_vram = f"{vram_bytes / (1024**3):.1f} GB"
        except ImportError:
            pass

        return {
            "status": "ok",
            "version": "0.1.0",
            "copernicus_credentials": settings.has_copernicus_credentials,
            "pytorch_available": pytorch_available,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "gpu_vram": gpu_vram,
        }

    # ── Web UI (Jinja2) ───────────────────────────────────────────────────────


    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request):
        """Serve the main dashboard."""
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "settings": {
                    "has_credentials": settings.has_copernicus_credentials,
                    "default_cloud": settings.default_max_cloud_coverage,
                    "default_window": settings.default_search_window_days,
                },
            },
        )

    return app


# Create the application instance
app = create_app()
