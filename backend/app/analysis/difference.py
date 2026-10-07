"""
Classical change detection using image differencing.

Implements a robust multi-step change detection pipeline:

1. Compute absolute normalized difference between before/after images.
2. Apply Gaussian smoothing (noise reduction).
3. Threshold to create binary change mask.
4. Apply morphological operations (remove noise, fill gaps).
5. Connected-component analysis (remove small regions).

Mathematical basis:
  For each band b at pixel (i, j):
    diff_b(i,j) = |A_b(i,j) - B_b(i,j)|

  The multi-band change score is a weighted average across bands:
    change_score = sum_b(w_b * norm(diff_b)) / sum_b(w_b)

  Where norm() maps each band difference to [0, 1] using its
  image-wide maximum (so bands with different numerical ranges
  contribute equally).

Scientific caution:
  A large pixel difference does NOT automatically indicate
  meaningful environmental change. Differences can arise from:
  - Clouds and cloud shadows (use cloud masking first)
  - Seasonal vegetation change (compare same-season images)
  - Illumination/sun angle differences
  - Atmospheric effects
  - Sensor noise and resampling artifacts
  The system explicitly labels all outputs as estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Default Gaussian sigma for noise reduction
DEFAULT_GAUSSIAN_SIGMA = 1.5

# Default change threshold (fraction of the normalized range)
DEFAULT_CHANGE_THRESHOLD = 0.15

# Morphological kernel size for opening/closing operations
DEFAULT_MORPH_KERNEL_SIZE = 3

# Minimum number of connected pixels to keep as a change region
DEFAULT_MIN_REGION_PIXELS = 50


@dataclass
class DifferenceResult:
    """Result of a change detection computation."""

    raw_difference: np.ndarray
    """Per-band absolute differences [n_bands, H, W] (normalized 0–1)."""

    change_score: np.ndarray
    """Combined multi-band change score [H, W] in range [0, 1]."""

    binary_mask: np.ndarray
    """Binary change mask [H, W]: 1=changed, 0=unchanged."""

    valid_mask: np.ndarray
    """Boolean mask [H, W]: True where both images have valid pixels."""

    threshold_used: float
    """The threshold value applied to change_score."""

    n_changed_pixels: int
    """Number of pixels classified as changed."""

    n_valid_pixels: int
    """Number of valid (non-masked) pixels."""

    @property
    def change_percentage(self) -> float:
        """Percentage of valid pixels classified as changed."""
        if self.n_valid_pixels == 0:
            return 0.0
        return 100.0 * self.n_changed_pixels / self.n_valid_pixels


def compute_difference(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    band_weights: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Compute normalized absolute difference between two images.

    For each band: diff_b = |before_b - after_b| / max(diff_b)

    Args:
        before:       Array [n_bands, H, W] (float32 reflectance).
        after:        Array [n_bands, H, W] (float32 reflectance).
        valid_mask:   Boolean [H, W]. If provided, only valid pixels
                      are used to compute normalization.
        band_weights: Dict mapping band index → weight multiplier.
                      Default: all bands have equal weight (1.0).

    Returns:
        Normalized difference array [n_bands, H, W] in range [0, 1].
        Masked (invalid) pixels are set to 0.
    """
    if before.shape != after.shape:
        raise ValueError(
            f"Shape mismatch: before={before.shape}, after={after.shape}"
        )

    n_bands = before.shape[0] if before.ndim == 3 else 1
    if before.ndim == 2:
        before = before[np.newaxis, ...]
        after = after[np.newaxis, ...]

    raw_diff = np.abs(before.astype(np.float32) - after.astype(np.float32))

    # Normalize each band independently so different value ranges
    # contribute equally to the combined score
    normalized = np.zeros_like(raw_diff)
    for b in range(n_bands):
        band_diff = raw_diff[b]
        if valid_mask is not None:
            valid_values = band_diff[valid_mask & np.isfinite(band_diff)]
        else:
            valid_values = band_diff[np.isfinite(band_diff)]

        if valid_values.size == 0:
            continue

        max_val = float(np.percentile(valid_values, 99))  # Use 99th percentile for robustness
        if max_val > 0:
            normalized[b] = np.clip(band_diff / max_val, 0.0, 1.0)

    # Apply band weights
    if band_weights:
        for b, w in band_weights.items():
            if b < n_bands:
                normalized[b] *= w

    # Zero out invalid pixels
    if valid_mask is not None:
        normalized[:, ~valid_mask] = 0.0

    return normalized


def compute_change_score(
    normalized_difference: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine per-band normalized differences into a single change score.

    The score is the mean across all bands:
      score = mean_b(diff_b)

    Args:
        normalized_difference: Array [n_bands, H, W] in [0, 1].
        valid_mask:           Boolean [H, W] mask.

    Returns:
        2D change score array [H, W] in range [0, 1].
    """
    score = np.mean(normalized_difference, axis=0)
    if valid_mask is not None:
        score[~valid_mask] = 0.0
    return score.astype(np.float32)


def apply_gaussian_smoothing(
    score: np.ndarray,
    sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Apply Gaussian smoothing to reduce salt-and-pepper noise.

    A moderate sigma (1–2 pixels) reduces isolated noise while
    preserving real change boundaries.

    Args:
        score: 2D change score array [H, W].
        sigma: Gaussian standard deviation in pixels.

    Returns:
        Smoothed score array [H, W].
    """
    return gaussian_filter(score, sigma=sigma).astype(np.float32)


def apply_threshold(
    score: np.ndarray,
    threshold: float = DEFAULT_CHANGE_THRESHOLD,
) -> np.ndarray:
    """
    Apply a threshold to produce a binary change mask.

    Pixels with score >= threshold are classified as changed (1).
    Pixels below threshold are classified as unchanged (0).

    Args:
        score:     2D change score [H, W] in [0, 1].
        threshold: Threshold value in [0, 1].

    Returns:
        Binary mask [H, W] with values 0 or 1, dtype uint8.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
    return (score >= threshold).astype(np.uint8)


def apply_morphological_cleanup(
    binary_mask: np.ndarray,
    kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
    min_region_pixels: int = DEFAULT_MIN_REGION_PIXELS,
) -> np.ndarray:
    """
    Apply morphological operations to clean up the binary change mask.

    Steps:
    1. Morphological opening (erosion + dilation): removes small noise spots.
    2. Morphological closing (dilation + erosion): fills small holes.
    3. Connected-component analysis: remove regions smaller than
       min_region_pixels.

    Args:
        binary_mask:       Binary mask [H, W] (uint8, 0 or 1).
        kernel_size:       Structuring element size (pixels).
        min_region_pixels: Minimum region size to keep.

    Returns:
        Cleaned binary mask [H, W] (uint8, 0 or 1).
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    # Opening removes thin noise
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Closing fills small holes in detected regions
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Remove small connected components
    cleaned = _remove_small_components(closed, min_region_pixels)

    n_before = int(binary_mask.sum())
    n_after = int(cleaned.sum())
    logger.debug(
        "Morphological cleanup: %d → %d changed pixels (removed %d noise pixels).",
        n_before,
        n_after,
        n_before - n_after,
    )
    return cleaned


def _remove_small_components(
    mask: np.ndarray,
    min_pixels: int,
) -> np.ndarray:
    """
    Remove connected components smaller than min_pixels.

    Uses scipy's label() for connected-component analysis.

    Args:
        mask:      Binary mask [H, W].
        min_pixels: Minimum number of pixels to keep.

    Returns:
        Filtered binary mask.
    """
    labeled, n_components = label(mask)
    if n_components == 0:
        return mask.copy()

    sizes = np.bincount(labeled.ravel())
    # Exclude component 0 (background)
    keep = np.zeros(n_components + 1, dtype=bool)
    keep[0] = False
    for comp_id in range(1, n_components + 1):
        if sizes[comp_id] >= min_pixels:
            keep[comp_id] = True

    cleaned = keep[labeled].astype(np.uint8)
    return cleaned


def detect_changes(
    before: np.ndarray,
    after: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    band_weights: Optional[Dict[int, float]] = None,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
    threshold: float = DEFAULT_CHANGE_THRESHOLD,
    morph_kernel_size: int = DEFAULT_MORPH_KERNEL_SIZE,
    min_region_pixels: int = DEFAULT_MIN_REGION_PIXELS,
) -> DifferenceResult:
    """
    Full classical change detection pipeline.

    Pipeline:
      compute_difference → change_score → gaussian_smoothing →
      threshold → morphological_cleanup

    Args:
        before:           Before image [n_bands, H, W] (float32 reflectance).
        after:            After image [n_bands, H, W] (float32 reflectance).
        valid_mask:       Boolean [H, W] mask (True = valid pixel).
        band_weights:     Optional per-band weights.
        gaussian_sigma:   Smoothing sigma in pixels.
        threshold:        Change detection threshold [0, 1].
        morph_kernel_size: Morphological operation kernel size.
        min_region_pixels: Minimum connected region size to keep.

    Returns:
        DifferenceResult with all intermediate and final outputs.
    """
    logger.info(
        "Running classical change detection | threshold=%.2f | sigma=%.1f",
        threshold,
        gaussian_sigma,
    )

    # Ensure we have a valid mask
    if valid_mask is None:
        valid_mask = (
            np.all(np.isfinite(before), axis=0) &
            np.all(np.isfinite(after), axis=0)
        )

    # Step 1: Per-band normalized difference
    raw_diff = compute_difference(before, after, valid_mask, band_weights)

    # Step 2: Combined change score
    score = compute_change_score(raw_diff, valid_mask)

    # Step 3: Gaussian smoothing
    smoothed = apply_gaussian_smoothing(score, sigma=gaussian_sigma)

    # Re-apply valid mask after smoothing
    smoothed[~valid_mask] = 0.0

    # Step 4: Threshold
    binary = apply_threshold(smoothed, threshold=threshold)

    # Step 5: Morphological cleanup
    cleaned = apply_morphological_cleanup(
        binary,
        kernel_size=morph_kernel_size,
        min_region_pixels=min_region_pixels,
    )

    # Only count changes within valid region
    cleaned[~valid_mask] = 0

    n_changed = int(cleaned.sum())
    n_valid = int(valid_mask.sum())

    result = DifferenceResult(
        raw_difference=raw_diff,
        change_score=smoothed,
        binary_mask=cleaned,
        valid_mask=valid_mask,
        threshold_used=threshold,
        n_changed_pixels=n_changed,
        n_valid_pixels=n_valid,
    )

    logger.info(
        "Change detection complete: %.1f%% of valid area shows significant change "
        "(%d / %d pixels).",
        result.change_percentage,
        n_changed,
        n_valid,
    )
    return result
