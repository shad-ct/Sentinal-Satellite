"""
Change statistics computation and area estimation.

Converts pixel-level change results into meaningful area measurements
and per-category breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.app.utils.geo import BoundingBox
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChangeStatistics:
    """
    Statistical summary of detected changes.

    All areas are approximate — they depend on the pixel resolution,
    bounding box, and the projection used for area calculation.
    Results are clearly labelled as estimates in the UI.
    """

    total_area_km2: float
    """Total analysed area in km²."""

    valid_area_km2: float
    """Valid (non-cloud, non-masked) area in km²."""

    changed_area_km2: float
    """Area classified as changed in km²."""

    change_percentage: float
    """Percentage of valid area that changed."""

    n_changed_pixels: int
    """Raw count of changed pixels."""

    n_valid_pixels: int
    """Raw count of valid pixels."""

    n_total_pixels: int
    """Total pixels in the image."""

    pixel_area_m2: float
    """Area of a single pixel in m²."""

    index_hints: Dict[str, float] = field(default_factory=dict)
    """
    Per-category area hints derived from spectral indices.
    These are ESTIMATES only — not confirmed classifications.

    Keys: 'vegetation_decrease_km2', 'vegetation_increase_km2',
          'water_change_km2', 'buildup_change_km2'.
    """

    cloud_coverage_pct: float = 0.0
    """Cloud coverage percentage of the full image."""

    notes: List[str] = field(default_factory=list)
    """Scientific caveats and uncertainty notes."""


def compute_statistics(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    bbox: BoundingBox,
    resolution_m: int = 10,
    index_changes: Optional[Dict[str, np.ndarray]] = None,
    cloud_coverage_pct: float = 0.0,
) -> ChangeStatistics:
    """
    Compute area-based change statistics.

    Args:
        binary_mask:      2D uint8 mask [H, W]: 1=changed, 0=unchanged.
        valid_mask:       2D boolean mask [H, W]: True=valid pixels.
        bbox:             Geographic bounding box of the analysed area.
        resolution_m:     Pixel size in metres (default 10 m).
        index_changes:    Dict mapping index name → 2D difference array.
                          Used to provide per-category hints.
        cloud_coverage_pct: Cloud coverage percentage.

    Returns:
        ChangeStatistics with all area measurements.
    """
    pixel_area_m2 = float(resolution_m ** 2)
    pixel_area_km2 = pixel_area_m2 / 1_000_000.0

    h, w = binary_mask.shape
    n_total = h * w
    n_valid = int(valid_mask.sum())
    n_changed = int((binary_mask & valid_mask).sum())

    total_km2 = n_total * pixel_area_km2
    valid_km2 = n_valid * pixel_area_km2
    changed_km2 = n_changed * pixel_area_km2
    change_pct = (100.0 * n_changed / n_valid) if n_valid > 0 else 0.0

    # Compute per-index category hints
    index_hints = {}
    if index_changes:
        change_region = (binary_mask == 1) & valid_mask
        index_hints = _compute_index_hints(
            index_changes=index_changes,
            change_region=change_region,
            pixel_area_km2=pixel_area_km2,
        )

    notes = [
        "All area estimates are based on pixel counting at the specified resolution.",
        f"Pixel size: {resolution_m} m × {resolution_m} m ({pixel_area_m2:.0f} m²).",
        "Index-based category hints are spectral estimates, not confirmed land-cover classifications.",
        "Seasonal variation, cloud shadows, and atmospheric effects can produce false detections.",
    ]

    stats = ChangeStatistics(
        total_area_km2=round(total_km2, 4),
        valid_area_km2=round(valid_km2, 4),
        changed_area_km2=round(changed_km2, 4),
        change_percentage=round(change_pct, 2),
        n_changed_pixels=n_changed,
        n_valid_pixels=n_valid,
        n_total_pixels=n_total,
        pixel_area_m2=pixel_area_m2,
        index_hints={k: round(v, 4) for k, v in index_hints.items()},
        cloud_coverage_pct=cloud_coverage_pct,
        notes=notes,
    )

    logger.info(
        "Statistics: total=%.2f km² | valid=%.2f km² | changed=%.2f km² (%.1f%%)",
        stats.total_area_km2,
        stats.valid_area_km2,
        stats.changed_area_km2,
        stats.change_percentage,
    )
    return stats


def _compute_index_hints(
    index_changes: Dict[str, np.ndarray],
    change_region: np.ndarray,
    pixel_area_km2: float,
) -> Dict[str, float]:
    """
    Estimate per-category areas within the detected change region
    using spectral index differences.

    These are HINTS only — not ground-truth classifications.
    Thresholds used here are conservative defaults.

    Args:
        index_changes:  Dict mapping index name → 2D difference array.
        change_region:  Boolean mask of changed pixels.
        pixel_area_km2: Area of one pixel in km².

    Returns:
        Dict with area estimates per category.
    """
    hints = {}

    # NDVI change hints
    if "NDVI_change" in index_changes:
        ndvi_diff = index_changes["NDVI_change"]
        # Possible vegetation decrease: NDVI dropped by > 0.1 in changed area
        veg_decrease = change_region & (ndvi_diff < -0.1) & np.isfinite(ndvi_diff)
        hints["vegetation_decrease_km2"] = float(veg_decrease.sum()) * pixel_area_km2

        # Possible vegetation increase: NDVI increased by > 0.1 in changed area
        veg_increase = change_region & (ndvi_diff > 0.1) & np.isfinite(ndvi_diff)
        hints["vegetation_increase_km2"] = float(veg_increase.sum()) * pixel_area_km2

    # NDWI change hints
    if "NDWI_change" in index_changes:
        ndwi_diff = index_changes["NDWI_change"]
        water_change = change_region & (np.abs(ndwi_diff) > 0.1) & np.isfinite(ndwi_diff)
        hints["water_change_km2"] = float(water_change.sum()) * pixel_area_km2

    # NDBI change hints
    if "NDBI_change" in index_changes:
        ndbi_diff = index_changes["NDBI_change"]
        buildup_change = change_region & (ndbi_diff > 0.1) & np.isfinite(ndbi_diff)
        hints["buildup_change_km2"] = float(buildup_change.sum()) * pixel_area_km2

    return hints
