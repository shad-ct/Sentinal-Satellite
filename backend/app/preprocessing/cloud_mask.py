"""
Cloud masking for Sentinel-2 imagery.

Uses the Scene Classification Layer (SCL) from Sentinel-2 L2A products
to generate a cloud mask distinguishing:
  - Valid pixels
  - Cloud pixels (thin + thick cloud, cirrus)
  - Cloud shadow pixels
  - No-data / saturated / unclassified pixels

The SCL band uses the following class codes:
  0 = No Data
  1 = Saturated or defective pixel
  2 = Dark Area Pixels
  3 = Cloud Shadows
  4 = Vegetation
  5 = Bare Soils
  6 = Water
  7 = Unclassified
  8 = Cloud medium probability
  9 = Cloud high probability
  10 = Thin cirrus
  11 = Snow or ice
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class SCLClass(IntEnum):
    """Sentinel-2 SCL classification codes."""
    NO_DATA = 0
    SATURATED = 1
    DARK_AREA = 2
    CLOUD_SHADOW = 3
    VEGETATION = 4
    BARE_SOIL = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM = 8
    CLOUD_HIGH = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


class MaskClass(IntEnum):
    """Output mask classification (simpler than SCL)."""
    VALID = 0
    CLOUD = 1
    CLOUD_SHADOW = 2
    INVALID = 3


# SCL codes considered as CLOUD pixels
CLOUD_SCL_CODES = {
    SCLClass.CLOUD_MEDIUM,
    SCLClass.CLOUD_HIGH,
    SCLClass.THIN_CIRRUS,
}

# SCL codes considered as CLOUD SHADOW
SHADOW_SCL_CODES = {
    SCLClass.CLOUD_SHADOW,
    SCLClass.DARK_AREA,
}

# SCL codes considered as INVALID (no data, saturated, etc.)
INVALID_SCL_CODES = {
    SCLClass.NO_DATA,
    SCLClass.SATURATED,
}

# SCL codes that can be included as valid land (user-configurable)
VALID_SCL_CODES = {
    SCLClass.VEGETATION,
    SCLClass.BARE_SOIL,
    SCLClass.WATER,
    SCLClass.UNCLASSIFIED,
    SCLClass.SNOW_ICE,
}


def create_cloud_mask(
    scl_band: np.ndarray,
    include_shadow: bool = True,
    include_cirrus: bool = True,
) -> np.ndarray:
    """
    Create a 4-class cloud mask from a Sentinel-2 SCL band array.

    Args:
        scl_band:       2D integer array with SCL class values.
        include_shadow: If True, cloud shadows are marked as CLOUD_SHADOW.
                        If False, dark areas are treated as VALID.
        include_cirrus: If True, thin cirrus (SCL=10) is marked as CLOUD.
                        If False, thin cirrus is treated as VALID.

    Returns:
        2D uint8 array with values from MaskClass:
          0 = VALID
          1 = CLOUD
          2 = CLOUD_SHADOW
          3 = INVALID

    Note:
        This function does NOT perform dilation of cloud edges.
        Cloud edges are inherently uncertain — consider eroding valid
        areas slightly for conservative masking.
    """
    scl = scl_band.astype(np.uint8)
    mask = np.full_like(scl, MaskClass.VALID, dtype=np.uint8)

    # Mark cloud pixels
    cloud_codes = set(CLOUD_SCL_CODES)
    if not include_cirrus:
        cloud_codes.discard(SCLClass.THIN_CIRRUS)

    for code in cloud_codes:
        mask[scl == code] = MaskClass.CLOUD

    # Mark cloud shadow pixels
    if include_shadow:
        for code in SHADOW_SCL_CODES:
            mask[scl == code] = MaskClass.CLOUD_SHADOW

    # Mark invalid pixels
    for code in INVALID_SCL_CODES:
        mask[scl == code] = MaskClass.INVALID

    return mask


def apply_cloud_mask(
    data: np.ndarray,
    cloud_mask: np.ndarray,
    fill_value: float = np.nan,
    mask_classes: Optional[set] = None,
) -> np.ndarray:
    """
    Apply a cloud mask to a multi-band image, filling masked pixels.

    Args:
        data:          Array of shape [bands, H, W] or [H, W].
        cloud_mask:    2D uint8 array with MaskClass values.
        fill_value:    Value to assign to masked pixels (default: NaN).
        mask_classes:  Set of MaskClass values to mask out.
                       Defaults to {CLOUD, CLOUD_SHADOW, INVALID}.

    Returns:
        Copy of data with masked pixels set to fill_value.
        Output dtype is always float32 (to support NaN).
    """
    if mask_classes is None:
        mask_classes = {MaskClass.CLOUD, MaskClass.CLOUD_SHADOW, MaskClass.INVALID}

    masked_pixels = np.zeros_like(cloud_mask, dtype=bool)
    for cls in mask_classes:
        masked_pixels |= (cloud_mask == cls)

    result = data.astype(np.float32).copy()

    if result.ndim == 2:
        result[masked_pixels] = fill_value
    elif result.ndim == 3:
        result[:, masked_pixels] = fill_value
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    return result


def calculate_valid_pixel_percentage(
    cloud_mask: np.ndarray,
    region_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate the percentage of pixels in each mask class.

    Args:
        cloud_mask:  2D cloud mask array (MaskClass values).
        region_mask: Optional boolean mask for a sub-region.
                     If provided, only pixels where region_mask=True are counted.

    Returns:
        Dict with percentage for each class and total valid %.
        Example:
          {
            "valid_pct": 78.3,
            "cloud_pct": 15.2,
            "shadow_pct": 4.1,
            "invalid_pct": 2.4,
            "usable_pct": 78.3,
          }
    """
    if region_mask is not None:
        mask_region = cloud_mask[region_mask]
    else:
        mask_region = cloud_mask.ravel()

    total = len(mask_region)
    if total == 0:
        return {
            "valid_pct": 0.0,
            "cloud_pct": 0.0,
            "shadow_pct": 0.0,
            "invalid_pct": 0.0,
            "usable_pct": 0.0,
            "total_pixels": 0,
        }

    counts = {cls: int(np.sum(mask_region == cls)) for cls in MaskClass}
    valid_count = counts[MaskClass.VALID]

    stats = {
        "valid_pct": 100.0 * valid_count / total,
        "cloud_pct": 100.0 * counts[MaskClass.CLOUD] / total,
        "shadow_pct": 100.0 * counts[MaskClass.CLOUD_SHADOW] / total,
        "invalid_pct": 100.0 * counts[MaskClass.INVALID] / total,
        "usable_pct": 100.0 * valid_count / total,
        "total_pixels": total,
    }

    logger.debug(
        "Cloud mask stats: valid=%.1f%% cloud=%.1f%% shadow=%.1f%% invalid=%.1f%%",
        stats["valid_pct"],
        stats["cloud_pct"],
        stats["shadow_pct"],
        stats["invalid_pct"],
    )
    return stats


def extract_scl_from_multiband(
    data: np.ndarray,
    band_names: list[str],
) -> Optional[np.ndarray]:
    """
    Extract the SCL band from a multi-band image array.

    Args:
        data:       Array of shape [n_bands, H, W].
        band_names: List of band names matching the array's band axis.

    Returns:
        2D SCL array, or None if SCL is not present.
    """
    if "SCL" not in band_names:
        logger.warning(
            "SCL band not found in band list %s. "
            "Cloud masking will not be applied.",
            band_names,
        )
        return None

    scl_idx = band_names.index("SCL")
    return data[scl_idx].astype(np.uint8)
