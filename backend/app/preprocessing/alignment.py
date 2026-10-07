"""
Geospatial image alignment for multi-temporal Sentinel-2 imagery.

Ensures that two images from different dates are perfectly aligned:
- Same CRS
- Same spatial extent (bounding box)
- Same resolution
- Same pixel grid (same width, height, transform)

This module uses rasterio's warp functionality to reproject images
onto a common reference grid. It does NOT use OpenCV resize, which
would ignore geospatial metadata and produce misaligned images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Default resampling method for alignment
# Bilinear is a good balance between quality and speed for continuous data.
# For classification bands (e.g., SCL), use Resampling.nearest.
DEFAULT_RESAMPLING = Resampling.bilinear
CLASSIFICATION_RESAMPLING = Resampling.nearest


def align_images(
    reference: np.ndarray,
    reference_profile: dict,
    target: np.ndarray,
    target_profile: dict,
    resampling: Resampling = DEFAULT_RESAMPLING,
) -> Tuple[np.ndarray, dict]:
    """
    Reproject and resample `target` to match the reference image grid.

    After alignment:
    - Both images have the same CRS.
    - Both images have the same width, height, and transform.
    - No information about the original spatial context is lost.

    Args:
        reference:         Reference image array [n_bands, H, W].
        reference_profile: Rasterio profile for the reference image.
        target:            Target image array to warp [n_bands, H, W].
        target_profile:    Rasterio profile for the target image.
        resampling:        Resampling method (default: bilinear).

    Returns:
        Tuple of (aligned_target array, updated profile matching reference).

    Raises:
        ValueError: If either image lacks CRS or transform information.
    """
    ref_crs = reference_profile.get("crs")
    tgt_crs = target_profile.get("crs")
    ref_transform = reference_profile.get("transform")
    tgt_transform = target_profile.get("transform")

    if ref_crs is None or ref_transform is None:
        raise ValueError("Reference image missing CRS or transform in profile.")
    if tgt_crs is None or tgt_transform is None:
        raise ValueError("Target image missing CRS or transform in profile.")

    ref_h = reference_profile["height"]
    ref_w = reference_profile["width"]
    n_bands = target.shape[0] if target.ndim == 3 else 1

    if target.ndim == 2:
        target = target[np.newaxis, ...]

    # Check if alignment is actually needed
    if _profiles_match(reference_profile, target_profile, ref_h, ref_w):
        logger.debug("Images already aligned — no reprojection needed.")
        aligned_profile = reference_profile.copy()
        aligned_profile["count"] = n_bands
        return target.copy(), aligned_profile

    logger.info(
        "Aligning images: CRS %s→%s | size (%dx%d)→(%dx%d)",
        tgt_crs,
        ref_crs,
        target_profile.get("width", "?"),
        target_profile.get("height", "?"),
        ref_w,
        ref_h,
    )

    aligned = np.zeros((n_bands, ref_h, ref_w), dtype=np.float32)

    for band_idx in range(n_bands):
        reproject(
            source=target[band_idx],
            destination=aligned[band_idx],
            src_transform=tgt_transform,
            src_crs=tgt_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
        )

    aligned_profile = reference_profile.copy()
    aligned_profile.update(
        count=n_bands,
        dtype=np.float32,
    )

    logger.info("Alignment complete: output shape = %s", aligned.shape)
    return aligned, aligned_profile


def align_images_from_files(
    reference_path: Union[str, Path],
    target_path: Union[str, Path],
    resampling: Resampling = DEFAULT_RESAMPLING,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load two GeoTIFF files and align the target to the reference.

    Args:
        reference_path: Path to the reference GeoTIFF.
        target_path:    Path to the GeoTIFF to align.
        resampling:     Resampling method.

    Returns:
        Tuple of (reference array, aligned target array, shared profile).
    """
    reference_path = Path(reference_path)
    target_path = Path(target_path)

    with rasterio.open(reference_path) as ref_src:
        reference = ref_src.read().astype(np.float32)
        ref_profile = ref_src.profile.copy()

    with rasterio.open(target_path) as tgt_src:
        target = tgt_src.read().astype(np.float32)
        tgt_profile = tgt_src.profile.copy()

    logger.info(
        "Aligning %s → %s",
        target_path.name,
        reference_path.name,
    )

    aligned_target, shared_profile = align_images(
        reference=reference,
        reference_profile=ref_profile,
        target=target,
        target_profile=tgt_profile,
        resampling=resampling,
    )

    return reference, aligned_target, shared_profile


def verify_alignment(
    image_a: np.ndarray,
    profile_a: dict,
    image_b: np.ndarray,
    profile_b: dict,
    tolerance: float = 1e-6,
) -> bool:
    """
    Verify that two images share the same spatial grid.

    Checks:
    1. Same CRS
    2. Same dimensions (width, height)
    3. Same transform (pixel size and origin)

    Args:
        image_a:   First image array.
        profile_a: Rasterio profile for image_a.
        image_b:   Second image array.
        profile_b: Rasterio profile for image_b.
        tolerance: Tolerance for transform comparison.

    Returns:
        True if all alignment checks pass.
    """
    issues = []

    # Check CRS
    crs_a = profile_a.get("crs")
    crs_b = profile_b.get("crs")
    if str(crs_a) != str(crs_b):
        issues.append(f"CRS mismatch: {crs_a} vs {crs_b}")

    # Check dimensions
    h_a = profile_a.get("height", image_a.shape[-2])
    w_a = profile_a.get("width", image_a.shape[-1])
    h_b = profile_b.get("height", image_b.shape[-2])
    w_b = profile_b.get("width", image_b.shape[-1])
    if h_a != h_b or w_a != w_b:
        issues.append(f"Dimension mismatch: ({w_a}×{h_a}) vs ({w_b}×{h_b})")

    # Check transform
    tf_a = profile_a.get("transform")
    tf_b = profile_b.get("transform")
    if tf_a is not None and tf_b is not None:
        params_a = list(tf_a)[:6]
        params_b = list(tf_b)[:6]
        if not all(abs(a - b) < tolerance for a, b in zip(params_a, params_b)):
            issues.append(
                f"Transform mismatch: {params_a} vs {params_b}"
            )

    if issues:
        for issue in issues:
            logger.warning("Alignment check FAILED: %s", issue)
        return False

    logger.debug("Alignment verified: images are on the same spatial grid.")
    return True


def _profiles_match(
    profile_a: dict,
    profile_b: dict,
    height: int,
    width: int,
) -> bool:
    """Check if two rasterio profiles describe the same spatial grid."""
    if profile_b.get("width") != width or profile_b.get("height") != height:
        return False
    if str(profile_a.get("crs")) != str(profile_b.get("crs")):
        return False
    tf_a = profile_a.get("transform")
    tf_b = profile_b.get("transform")
    if tf_a is None or tf_b is None:
        return False
    return list(tf_a)[:6] == list(tf_b)[:6]
