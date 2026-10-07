"""
Radiometric normalization for Sentinel-2 imagery.

Normalizes band DN values to surface reflectance [0, 1] and applies
optional per-image histogram matching to reduce illumination differences
between multi-temporal image pairs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Sentinel-2 L2A standard reflectance scale
S2_SCALE_FACTOR = 10_000.0

# Typical valid reflectance range for S2 L2A after scaling
S2_VALID_REFLECTANCE_MIN = 0.0
S2_VALID_REFLECTANCE_MAX = 1.0


def normalize_s2_reflectance(
    band: np.ndarray,
    scale_factor: float = S2_SCALE_FACTOR,
    clip: bool = True,
) -> np.ndarray:
    """
    Convert Sentinel-2 L2A DN values to surface reflectance [0, 1].

    S2 L2A products store reflectance as uint16 with a 1/10000 scale factor.
    Dividing by 10000 gives approximate surface reflectance.

    Args:
        band:         Input array (uint16 DN values).
        scale_factor: Division factor (default 10000 for S2 L2A).
        clip:         If True, clip result to [0, 1].

    Returns:
        Float32 array in range [0, 1].
    """
    result = band.astype(np.float32) / scale_factor
    if clip:
        result = np.clip(result, S2_VALID_REFLECTANCE_MIN, S2_VALID_REFLECTANCE_MAX)
    return result


def normalize_band_stack(
    data: np.ndarray,
    band_names: list[str],
    scl_index: Optional[int] = None,
    scale_factor: float = S2_SCALE_FACTOR,
) -> np.ndarray:
    """
    Normalize a full multi-band stack.

    Applies reflectance normalization to all bands except:
    - SCL (Scene Classification Layer): kept as integer class codes.

    Args:
        data:        Array of shape [n_bands, H, W].
        band_names:  List of band names matching the band axis.
        scl_index:   Index of the SCL band (not normalized).
                     If None, derived from band_names.
        scale_factor: Division factor for reflectance conversion.

    Returns:
        Float32 array [n_bands, H, W] with reflectance in [0, 1],
        SCL band preserved as-is.
    """
    if scl_index is None and "SCL" in band_names:
        scl_index = band_names.index("SCL")

    result = np.zeros_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        if i == scl_index:
            result[i] = data[i].astype(np.float32)  # Keep SCL as codes
        else:
            result[i] = normalize_s2_reflectance(data[i], scale_factor=scale_factor)

    return result


def histogram_match(
    source: np.ndarray,
    reference: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Match the histogram of source to match the reference image.

    This reduces illumination and atmospheric differences between
    image pairs from different dates. Applied per-band.

    Note:
        This is an optional, additional normalization step.
        It reduces but does not eliminate seasonal/illumination differences.
        Use with caution — aggressive histogram matching can introduce
        spurious changes or hide real ones.

    Args:
        source:    Source image to modify [n_bands, H, W] or [H, W].
        reference: Reference image [n_bands, H, W] or [H, W].
        mask:      Optional boolean mask; if provided, only use valid
                   (True) pixels for histogram computation.

    Returns:
        Histogram-matched version of source (same shape, float32).
    """
    if source.shape != reference.shape:
        raise ValueError(
            f"source and reference must have same shape: "
            f"{source.shape} vs {reference.shape}"
        )

    is_2d = source.ndim == 2
    if is_2d:
        source = source[np.newaxis, ...]
        reference = reference[np.newaxis, ...]

    n_bands = source.shape[0]
    matched = np.zeros_like(source, dtype=np.float32)

    for b in range(n_bands):
        src_band = source[b].astype(np.float32)
        ref_band = reference[b].astype(np.float32)

        if mask is not None:
            src_valid = src_band[mask]
            ref_valid = ref_band[mask]
        else:
            src_valid = src_band.ravel()
            ref_valid = ref_band.ravel()

        # Remove NaN/inf
        src_valid = src_valid[np.isfinite(src_valid)]
        ref_valid = ref_valid[np.isfinite(ref_valid)]

        if src_valid.size < 100 or ref_valid.size < 100:
            logger.warning(
                "Band %d: too few valid pixels for histogram matching "
                "(src=%d, ref=%d). Skipping this band.",
                b,
                src_valid.size,
                ref_valid.size,
            )
            matched[b] = src_band
            continue

        matched[b] = _match_histogram_1d(src_band, src_valid, ref_valid)

    if is_2d:
        return matched[0]
    return matched


def _match_histogram_1d(
    source_band: np.ndarray,
    source_valid: np.ndarray,
    reference_valid: np.ndarray,
    n_bins: int = 1000,
) -> np.ndarray:
    """
    Apply 1D histogram matching to a single band.

    Uses CDF-based histogram equalization to map source pixel values
    to match the reference distribution.

    Args:
        source_band:    2D float32 source band.
        source_valid:   1D valid source pixel values (for CDF computation).
        reference_valid: 1D valid reference pixel values.
        n_bins:         Number of histogram bins.

    Returns:
        Histogram-matched source band (same shape, float32).
    """
    # Compute CDFs
    src_hist, src_edges = np.histogram(source_valid, bins=n_bins, density=True)
    ref_hist, ref_edges = np.histogram(reference_valid, bins=n_bins, density=True)

    src_cdf = np.cumsum(src_hist * np.diff(src_edges))
    ref_cdf = np.cumsum(ref_hist * np.diff(ref_edges))

    src_cdf = np.clip(src_cdf, 0, 1)
    ref_cdf = np.clip(ref_cdf, 0, 1)

    src_centers = (src_edges[:-1] + src_edges[1:]) / 2
    ref_centers = (ref_edges[:-1] + ref_edges[1:]) / 2

    # Map each source CDF value to the corresponding reference value
    matched_values = np.interp(source_band, src_centers, src_cdf)
    matched_band = np.interp(matched_values, ref_cdf, ref_centers)

    return matched_band.astype(np.float32)


def clip_percentile(
    data: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Clip data to the given percentile range (e.g., remove outliers).

    Args:
        data:  Input array.
        low:   Lower percentile (e.g., 2.0).
        high:  Upper percentile (e.g., 98.0).
        mask:  Boolean mask selecting valid pixels for percentile computation.

    Returns:
        Clipped array (same shape, float32).
    """
    flat = data[mask].ravel() if mask is not None else data.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return data.astype(np.float32)

    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    return np.clip(data, lo, hi).astype(np.float32)
