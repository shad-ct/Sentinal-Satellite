"""
FastAPI router for change detection analysis endpoints.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from backend.app.analysis.change_map import generate_visualizations
from backend.app.analysis.difference import detect_changes
from backend.app.analysis.indices import compute_all_indices, compute_index_change, get_index
from backend.app.analysis.statistics import compute_statistics
from backend.app.preprocessing.alignment import align_images_from_files
from backend.app.preprocessing.cloud_mask import (
    apply_cloud_mask,
    calculate_valid_pixel_percentage,
    create_cloud_mask,
    extract_scl_from_multiband,
)
from backend.app.preprocessing.normalization import normalize_band_stack
from backend.app.storage.database import get_db_session
from backend.app.storage.models import AnalysisResultRecord, ObservationRecord
from backend.app.storage.repository import (
    AnalysisPairRepository,
    AnalysisResultRepository,
    ObservationRepository,
)
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class CompareRequest(BaseModel):
    before_product_id: str
    after_product_id: str
    lat: float
    lon: float
    size_km: float = 10.0
    threshold: float = Field(0.15, ge=0.01, le=0.99)
    gaussian_sigma: float = Field(1.5, ge=0.0, le=10.0)
    min_region_pixels: int = Field(50, ge=1)
    bbox_west: Optional[float] = None
    bbox_south: Optional[float] = None
    bbox_east: Optional[float] = None
    bbox_north: Optional[float] = None

    def get_bbox(self) -> BoundingBox:
        if all(v is not None for v in [self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north]):
            return BoundingBox(self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north)
        return BoundingBox.from_center(self.lat, self.lon, self.size_km)


@router.post("/analysis/compare")
async def compare_images(req: CompareRequest, background_tasks: BackgroundTasks):
    """
    Run change detection between two downloaded observations.

    Both observations must already be downloaded via /api/imagery/download.
    Returns a result ID; poll /api/analysis/{id} for status.
    """
    session = get_db_session()
    try:
        obs_repo = ObservationRepository(session)
        before_rec = obs_repo.get_by_product_id(req.before_product_id)
        after_rec = obs_repo.get_by_product_id(req.after_product_id)
    finally:
        session.close()

    if before_rec is None:
        raise HTTPException(404, f"Before observation {req.before_product_id} not found. Download it first.")
    if after_rec is None:
        raise HTTPException(404, f"After observation {req.after_product_id} not found. Download it first.")

    if not before_rec.downloaded or not before_rec.local_path:
        raise HTTPException(400, "Before image not downloaded yet.")
    if not after_rec.downloaded or not after_rec.local_path:
        raise HTTPException(400, "After image not downloaded yet.")

    bbox = req.get_bbox()

    # Create DB records
    session = get_db_session()
    try:
        pair_repo = AnalysisPairRepository(session)
        result_repo = AnalysisResultRepository(session)

        pair = pair_repo.create(
            before_id=before_rec.id,
            after_id=after_rec.id,
            before_date=before_rec.acquisition_date,
            after_date=after_rec.acquisition_date,
            aoi_bbox=bbox.to_tuple(),
        )

        config = {
            "threshold": req.threshold,
            "gaussian_sigma": req.gaussian_sigma,
            "min_region_pixels": req.min_region_pixels,
            "before_product_id": req.before_product_id,
            "after_product_id": req.after_product_id,
            "software_version": "0.1.0",
        }
        result_record = result_repo.create(pair_id=pair.id, config=config)
        result_id = result_record.id
    finally:
        session.close()

    # Run analysis in background
    background_tasks.add_task(
        _run_analysis,
        result_id=result_id,
        before_path=Path(before_rec.local_path),
        after_path=Path(after_rec.local_path),
        before_bands=before_rec.get_bands(),
        after_bands=after_rec.get_bands(),
        bbox=bbox,
        threshold=req.threshold,
        gaussian_sigma=req.gaussian_sigma,
        min_region_pixels=req.min_region_pixels,
    )

    return {
        "result_id": result_id,
        "status": "running",
        "message": f"Analysis started. Poll /api/analysis/{result_id} for status.",
    }


@router.get("/analysis/{result_id}")
async def get_analysis_status(result_id: int):
    """Get the status and basic info of an analysis."""
    session = get_db_session()
    try:
        result_repo = AnalysisResultRepository(session)
        record = result_repo.get_by_id(result_id)
        if record is None:
            raise HTTPException(404, f"Analysis {result_id} not found.")

        return {
            "result_id": record.id,
            "pair_id": record.pair_id,
            "status": record.status,
            "error_message": record.error_message,
            "change_percentage": record.change_percentage,
            "changed_area_km2": record.changed_area_km2,
            "valid_area_km2": record.valid_area_km2,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "completed_at": record.completed_at.isoformat() if record.completed_at else None,
        }
    finally:
        session.close()


@router.get("/analysis/{result_id}/results")
async def get_analysis_results(result_id: int):
    """Get full results including paths to visualization files."""
    session = get_db_session()
    try:
        result_repo = AnalysisResultRepository(session)
        record = result_repo.get_by_id(result_id)
        if record is None:
            raise HTTPException(404, f"Analysis {result_id} not found.")

        settings = get_settings()
        output_paths = record.get_output_paths()

        # Convert absolute paths to URLs
        url_paths = {}
        for key, path_str in output_paths.items():
            p = Path(path_str)
            if p.exists():
                try:
                    rel = p.relative_to(settings.data_dir)
                    url_paths[key] = f"/data/{rel}"
                except ValueError:
                    url_paths[key] = path_str

        return {
            "result_id": record.id,
            "status": record.status,
            "error_message": record.error_message,
            "statistics": {
                "change_percentage": record.change_percentage,
                "changed_area_km2": record.changed_area_km2,
                "valid_area_km2": record.valid_area_km2,
                "total_area_km2": record.total_area_km2,
                "n_changed_pixels": record.n_changed_pixels,
                "n_valid_pixels": record.n_valid_pixels,
                "index_hints": record.get_index_hints(),
            },
            "visualizations": url_paths,
            "config": record.get_analysis_config(),
        }
    finally:
        session.close()


def _run_analysis(
    result_id: int,
    before_path: Path,
    after_path: Path,
    before_bands: list,
    after_bands: list,
    bbox: BoundingBox,
    threshold: float,
    gaussian_sigma: float,
    min_region_pixels: int,
) -> None:
    """
    Background task: full change detection pipeline.

    Steps:
    1. Read both GeoTIFFs
    2. Normalize to reflectance
    3. Extract cloud masks (SCL)
    4. Align target to reference
    5. Compute spectral indices (before + after)
    6. Run classical change detection
    7. Compute statistics
    8. Generate visualizations
    9. Save results to DB
    """
    session = get_db_session()
    result_repo = AnalysisResultRepository(session)
    result_repo.update_status(result_id, "running")
    session.close()

    try:
        logger.info("=== Change Detection Analysis #%d ===", result_id)

        # ── 1. Read images ──────────────────────────────────────────────────
        with rasterio.open(before_path) as src:
            before_data = src.read().astype(np.float32)
            before_profile = src.profile.copy()

        with rasterio.open(after_path) as src:
            after_data = src.read().astype(np.float32)
            after_profile = src.profile.copy()

        logger.info("Before shape: %s  After shape: %s", before_data.shape, after_data.shape)

        # ── 2. Normalize reflectance ────────────────────────────────────────
        before_norm = normalize_band_stack(before_data, before_bands)
        after_norm = normalize_band_stack(after_data, after_bands)

        # ── 3. Cloud masks ─────────────────────────────────────────────────
        before_scl = extract_scl_from_multiband(before_norm, before_bands)
        after_scl = extract_scl_from_multiband(after_norm, after_bands)

        before_cloud_mask = create_cloud_mask(before_scl) if before_scl is not None else None
        after_cloud_mask = create_cloud_mask(after_scl) if after_scl is not None else None

        # Compute valid mask: pixel must be valid in BOTH images
        from backend.app.preprocessing.cloud_mask import MaskClass
        if before_cloud_mask is not None and after_cloud_mask is not None:
            valid_mask = (before_cloud_mask == MaskClass.VALID) & (after_cloud_mask == MaskClass.VALID)
        elif before_cloud_mask is not None:
            valid_mask = before_cloud_mask == MaskClass.VALID
        elif after_cloud_mask is not None:
            valid_mask = after_cloud_mask == MaskClass.VALID
        else:
            valid_mask = np.ones(before_norm.shape[1:], dtype=bool)

        cloud_pct = 100.0 * (1.0 - valid_mask.mean())

        # ── 4. Align images ────────────────────────────────────────────────
        logger.info("Aligning images...")
        before_aligned, after_aligned, shared_profile = align_images_from_files(
            before_path, after_path
        )

        # Re-normalize aligned versions (align_images_from_files reads raw)
        before_final = normalize_band_stack(before_aligned, before_bands)
        after_final = normalize_band_stack(after_aligned, after_bands)

        # Ensure valid mask matches aligned dimensions
        if valid_mask.shape != before_final.shape[1:]:
            import cv2
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (before_final.shape[2], before_final.shape[1]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # ── 5. Spectral indices ────────────────────────────────────────────
        logger.info("Computing spectral indices...")
        # Filter out SCL from index computation
        data_bands_before = [b for b in before_bands if b != "SCL"]
        data_bands_after = [b for b in after_bands if b != "SCL"]

        # Get band indices (excluding SCL)
        scl_i_b = before_bands.index("SCL") if "SCL" in before_bands else None
        scl_i_a = after_bands.index("SCL") if "SCL" in after_bands else None

        b_data_idx = [i for i, b in enumerate(before_bands) if b != "SCL"]
        a_data_idx = [i for i, a in enumerate(after_bands) if a != "SCL"]

        before_for_idx = before_final[b_data_idx]
        after_for_idx = after_final[a_data_idx]

        before_indices = compute_all_indices(before_for_idx, data_bands_before)
        after_indices = compute_all_indices(after_for_idx, data_bands_after)

        # Compute per-index change
        index_changes = {}
        all_index_results = {}
        for idx_name, before_ir in before_indices.items():
            all_index_results[idx_name] = before_ir
            if idx_name in after_indices:
                after_ir = after_indices[idx_name]
                all_index_results[f"{idx_name}_after"] = after_ir
                change_ir = compute_index_change(before_ir, after_ir)
                all_index_results[f"{idx_name}_change"] = change_ir
                index_changes[f"{idx_name}_change"] = change_ir.values

        # ── 6. Change detection ────────────────────────────────────────────
        logger.info("Running classical change detection...")
        diff_result = detect_changes(
            before=before_for_idx,
            after=after_for_idx,
            valid_mask=valid_mask if valid_mask.shape == before_for_idx.shape[1:] else None,
            threshold=threshold,
            gaussian_sigma=gaussian_sigma,
            min_region_pixels=min_region_pixels,
        )

        # ── 7. Statistics ──────────────────────────────────────────────────
        stats = compute_statistics(
            binary_mask=diff_result.binary_mask,
            valid_mask=diff_result.valid_mask,
            bbox=bbox,
            resolution_m=shared_profile.get("res", [10])[0] if "res" in shared_profile else 10,
            index_changes=index_changes,
            cloud_coverage_pct=cloud_pct,
        )

        # ── 8. Visualizations ──────────────────────────────────────────────
        logger.info("Generating visualizations...")
        settings = get_settings()
        output_dir = settings.data_dir / "results" / f"analysis_{result_id}"

        vis_paths = generate_visualizations(
            before=before_for_idx,
            after=after_for_idx,
            change_score=diff_result.change_score,
            binary_mask=diff_result.binary_mask,
            valid_mask=diff_result.valid_mask,
            output_dir=output_dir,
            band_names=data_bands_before,
            index_results=all_index_results,
        )

        paths_dict = {k: str(v) for k, v in vis_paths.items()}

        # ── 9. Save to DB ──────────────────────────────────────────────────
        session = get_db_session()
        result_repo = AnalysisResultRepository(session)
        result_repo.update_results(
            result_id=result_id,
            stats={
                "change_percentage": stats.change_percentage,
                "changed_area_km2": stats.changed_area_km2,
                "valid_area_km2": stats.valid_area_km2,
                "total_area_km2": stats.total_area_km2,
                "n_changed_pixels": stats.n_changed_pixels,
                "n_valid_pixels": stats.n_valid_pixels,
                "index_hints": stats.index_hints,
            },
            paths=paths_dict,
        )
        session.close()

        logger.info(
            "=== Analysis #%d complete: %.1f%% change ===",
            result_id,
            stats.change_percentage,
        )

    except Exception as exc:
        logger.error("Analysis #%d failed: %s", result_id, exc, exc_info=True)
        session = get_db_session()
        result_repo = AnalysisResultRepository(session)
        result_repo.update_status(result_id, "error", error=str(exc))
        session.close()
