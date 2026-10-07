"""
FastAPI router for spectral index computation endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.analysis.indices import (
    compute_all_indices,
    compute_index_change,
    get_index,
    list_indices,
)
from backend.app.preprocessing.normalization import normalize_band_stack
from backend.app.utils.config import get_settings
from backend.app.utils.image import save_colormap_png
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class IndexRequest(BaseModel):
    image_path: str = Field(..., description="Absolute path to a downloaded GeoTIFF")
    band_names: List[str] = Field(..., description="Band names matching the GeoTIFF bands")
    index_name: Optional[str] = Field(None, description="Specific index (e.g. 'NDVI'). Omit for all.")


@router.get("/indices")
async def list_available_indices():
    """List all registered spectral indices."""
    indices = []
    for name in list_indices():
        idx = get_index(name)
        indices.append({
            "name": name,
            "description": idx.description,
            "required_bands": idx.required_bands,
            "positive_interpretation": idx.positive_interpretation,
            "negative_interpretation": idx.negative_interpretation,
        })
    return {"indices": indices}


@router.post("/indices/compute")
async def compute_index(req: IndexRequest):
    """Compute spectral indices for a downloaded image."""
    img_path = Path(req.image_path)
    if not img_path.exists():
        raise HTTPException(404, f"Image not found: {img_path}")

    try:
        with rasterio.open(img_path) as src:
            data = src.read().astype(np.float32)
    except Exception as exc:
        raise HTTPException(500, f"Failed to read image: {exc}")

    # Normalize to reflectance
    normed = normalize_band_stack(data, req.band_names)

    # Filter out SCL
    data_bands = [b for b in req.band_names if b != "SCL"]
    data_idx = [i for i, b in enumerate(req.band_names) if b != "SCL"]
    data_only = normed[data_idx]

    try:
        if req.index_name:
            idx_obj = get_index(req.index_name)
            results = {req.index_name: idx_obj.compute_from_stack(data_only, data_bands)}
        else:
            results = compute_all_indices(data_only, data_bands)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    output = {}
    for name, result in results.items():
        valid = result.valid_mask
        vals = result.values[valid]
        output[name] = {
            "name": name,
            "description": result.description,
            "min": float(np.nanmin(vals)) if vals.size > 0 else None,
            "max": float(np.nanmax(vals)) if vals.size > 0 else None,
            "mean": float(np.nanmean(vals)) if vals.size > 0 else None,
            "std": float(np.nanstd(vals)) if vals.size > 0 else None,
            "valid_pixel_pct": float(100.0 * valid.sum() / valid.size),
        }

    return {"image": req.image_path, "indices": output}


@router.post("/indices/ndvi")
async def compute_ndvi(req: IndexRequest):
    """Convenience endpoint to compute NDVI for an image."""
    req.index_name = "NDVI"
    return await compute_index(req)
