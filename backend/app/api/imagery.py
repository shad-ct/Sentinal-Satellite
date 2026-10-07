"""
FastAPI router for satellite imagery search and download endpoints.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from backend.app.satellite.catalog import CDSECatalog
from backend.app.satellite.downloader import DEFAULT_BANDS, Sentinel2Downloader
from backend.app.satellite.provider import Sentinel2Provider
from backend.app.storage.database import get_db_session
from backend.app.storage.models import ObservationRecord
from backend.app.storage.repository import ObservationRepository
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox, validate_coordinates
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Request / Response Schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Centre latitude")
    lon: float = Field(..., ge=-180, le=180, description="Centre longitude")
    start_date: date = Field(..., description="Search start date (YYYY-MM-DD)")
    end_date: date = Field(..., description="Search end date (YYYY-MM-DD)")
    max_cloud_coverage: float = Field(20.0, ge=0, le=100)
    size_km: float = Field(10.0, ge=0.5, le=200, description="AOI side length in km")
    # OR explicit bbox
    bbox_west: Optional[float] = None
    bbox_south: Optional[float] = None
    bbox_east: Optional[float] = None
    bbox_north: Optional[float] = None

    @model_validator(mode="after")
    def validate_dates(self) -> "SearchRequest":
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        return self

    def get_bbox(self) -> BoundingBox:
        if all(v is not None for v in [self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north]):
            return BoundingBox(self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north)
        return BoundingBox.from_center(self.lat, self.lon, self.size_km)


class ObservationResponse(BaseModel):
    product_id: str
    name: str
    acquisition_date: str
    platform: str
    cloud_coverage: float
    online: bool
    bbox: dict
    size_mb: Optional[float] = None


class DownloadRequest(BaseModel):
    product_id: str
    lat: float
    lon: float
    size_km: float = 10.0
    bands: List[str] = Field(default_factory=lambda: list(DEFAULT_BANDS))
    resolution_m: int = Field(10, ge=10, le=60)
    force_download: bool = False
    bbox_west: Optional[float] = None
    bbox_south: Optional[float] = None
    bbox_east: Optional[float] = None
    bbox_north: Optional[float] = None

    def get_bbox(self) -> BoundingBox:
        if all(v is not None for v in [self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north]):
            return BoundingBox(self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north)
        return BoundingBox.from_center(self.lat, self.lon, self.size_km)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/observations/search", response_model=List[ObservationResponse])
async def search_observations(req: SearchRequest):
    """
    Search for Sentinel-2 L2A observations within a bounding box and date range.

    Returns a list of available observations sorted by cloud coverage (best first).
    """
    settings = get_settings()
    try:
        settings.require_copernicus_credentials()
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    try:
        bbox = req.get_bbox()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        catalog = CDSECatalog()
        observations = catalog.search(
            bbox=bbox,
            start_date=req.start_date,
            end_date=req.end_date,
            max_cloud_coverage=req.max_cloud_coverage,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Optionally cache in DB
    if observations:
        try:
            session = get_db_session()
            repo = ObservationRepository(session)
            for obs in observations:
                repo.create_or_update(
                    product_id=obs.product_id,
                    name=obs.name,
                    acquisition_date=obs.acquisition_date.isoformat(),
                    acquisition_datetime=obs.acquisition_datetime.isoformat(),
                    platform=obs.platform,
                    cloud_coverage=obs.cloud_coverage,
                    bbox_west=obs.bbox.west,
                    bbox_south=obs.bbox.south,
                    bbox_east=obs.bbox.east,
                    bbox_north=obs.bbox.north,
                    online=obs.online,
                    size_mb=obs.size_mb,
                )
            session.close()
        except Exception as exc:
            logger.warning("Failed to cache observations in DB: %s", exc)

    return [
        ObservationResponse(
            product_id=obs.product_id,
            name=obs.name,
            acquisition_date=obs.acquisition_date.isoformat(),
            platform=obs.platform,
            cloud_coverage=obs.cloud_coverage,
            online=obs.online,
            bbox={
                "west": obs.bbox.west,
                "south": obs.bbox.south,
                "east": obs.bbox.east,
                "north": obs.bbox.north,
            },
            size_mb=obs.size_mb,
        )
        for obs in observations
    ]


@router.get("/observations")
async def list_downloaded_observations():
    """List all observations that have been downloaded locally."""
    session = get_db_session()
    try:
        repo = ObservationRepository(session)
        records = repo.list_downloaded()
        return [
            {
                "id": r.id,
                "product_id": r.product_id,
                "name": r.name,
                "acquisition_date": r.acquisition_date,
                "platform": r.platform,
                "cloud_coverage": r.cloud_coverage,
                "local_path": r.local_path,
                "bands": r.get_bands(),
            }
            for r in records
        ]
    finally:
        session.close()


@router.post("/imagery/download")
async def download_imagery(req: DownloadRequest):
    """
    Download Sentinel-2 imagery for a specific observation product ID.

    The imagery is saved locally as a GeoTIFF + metadata.json.
    Returns the local file path and image metadata.
    """
    settings = get_settings()
    try:
        settings.require_copernicus_credentials()
    except RuntimeError as exc:
        raise HTTPException(status_code=401, detail=str(exc))

    try:
        bbox = req.get_bbox()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Find the observation metadata
    session = get_db_session()
    try:
        repo = ObservationRepository(session)
        obs_record = repo.get_by_product_id(req.product_id)
    finally:
        session.close()

    if obs_record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Observation {req.product_id} not found. Run a search first.",
        )

    # Reconstruct Observation object
    from backend.app.satellite.catalog import Observation
    from datetime import datetime
    obs = Observation(
        product_id=obs_record.product_id,
        name=obs_record.name,
        acquisition_date=date.fromisoformat(obs_record.acquisition_date),
        acquisition_datetime=datetime.fromisoformat(obs_record.acquisition_datetime),
        platform=obs_record.platform,
        cloud_coverage=obs_record.cloud_coverage,
        bbox=bbox,
        online=obs_record.online,
    )

    try:
        downloader = Sentinel2Downloader()
        result = downloader.download(
            observation=obs,
            bbox=bbox,
            bands=req.bands,
            resolution_m=req.resolution_m,
            force_download=req.force_download,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Download failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Download error: {exc}")

    # Update DB record
    session = get_db_session()
    try:
        repo = ObservationRepository(session)
        repo.create_or_update(
            product_id=obs.product_id,
            name=obs.name,
            acquisition_date=obs.acquisition_date.isoformat(),
            acquisition_datetime=obs.acquisition_datetime.isoformat(),
            platform=obs.platform,
            cloud_coverage=obs.cloud_coverage,
            downloaded=True,
            local_path=str(result.local_path),
            bands=json.dumps(result.bands),
        )
    finally:
        session.close()

    # Build preview URL
    preview_path = result.local_path.parent / "rgb_preview.png"
    preview_url = None
    if preview_path.exists():
        rel = preview_path.relative_to(settings.data_dir)
        preview_url = f"/data/{rel}"

    return {
        "success": True,
        "product_id": result.product_id,
        "local_path": str(result.local_path),
        "preview_url": preview_url,
        "bands": result.bands,
        "acquisition_date": result.acquisition_date,
        "cloud_coverage": result.cloud_coverage,
        "resolution_m": result.resolution_m,
        "width": result.width,
        "height": result.height,
        "cached": result.cached,
        "bbox": {
            "west": result.bbox[0],
            "south": result.bbox[1],
            "east": result.bbox[2],
            "north": result.bbox[3],
        },
    }
