"""
Spectral index calculations for Sentinel-2 imagery.

Implements an extensible registry of spectral indices.
Each index is defined by its formula, required bands, and interpretation.

Implemented indices:
- NDVI  (Normalized Difference Vegetation Index)
- NDWI  (Normalized Difference Water Index)
- NDBI  (Normalized Difference Built-up Index)

Mathematical note:
  All normalized difference indices follow the form:
    NDI = (A - B) / (A + B)
  where A and B are reflectance values in different bands.
  The denominator is protected from division by zero by adding a small
  epsilon (1e-10) before division. Results are clipped to [-1, 1].

Caution on interpretation:
  Index values indicate SPECTRAL properties of the surface.
  They do NOT directly correspond to land-cover types without
  additional classification. Use hedged language in the UI:
  "possible vegetation" rather than "confirmed forest".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import numpy as np

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Small constant to avoid division by zero in normalised ratios
EPSILON = 1e-10

# Sentinel-2 band names for common wavelengths
BAND_RED = "B04"      # Red (665 nm)
BAND_GREEN = "B03"    # Green (560 nm)
BAND_BLUE = "B02"     # Blue (490 nm)
BAND_NIR = "B08"      # NIR wide (842 nm) — 10 m native resolution
BAND_SWIR1 = "B11"    # SWIR1 (1610 nm) — 20 m native resolution
BAND_SWIR2 = "B12"    # SWIR2 (2190 nm) — 20 m native resolution


@dataclass
class IndexResult:
    """Result of computing a spectral index."""

    name: str
    """Short name of the index (e.g. 'NDVI')."""

    values: np.ndarray
    """2D float32 array of index values in range [-1, 1] or [0, 1]."""

    valid_mask: np.ndarray
    """2D boolean array: True where the index is valid (both bands non-NaN)."""

    description: str = ""
    """Human-readable description of what the index measures."""

    positive_interpretation: str = ""
    """What a high positive value suggests (e.g. 'dense vegetation')."""

    negative_interpretation: str = ""
    """What a large negative value suggests (e.g. 'water or bare soil')."""


class SpectralIndex(ABC):
    """
    Abstract base class for spectral indices.

    Subclasses must define:
    - required_bands: list of band names needed
    - compute(): the actual formula
    """

    name: str = ""
    description: str = ""
    required_bands: List[str] = []
    positive_interpretation: str = ""
    negative_interpretation: str = ""

    def compute_from_dict(
        self,
        bands: Dict[str, np.ndarray],
    ) -> IndexResult:
        """
        Compute the index from a dict mapping band name → 2D array.

        Args:
            bands: Dict with band name keys and 2D float32 array values.
                   Arrays must be reflectance values (not raw DN).

        Returns:
            IndexResult with computed values and valid mask.

        Raises:
            KeyError: If a required band is missing from `bands`.
        """
        missing = [b for b in self.required_bands if b not in bands]
        if missing:
            raise KeyError(
                f"{self.name} requires bands {self.required_bands}, "
                f"but {missing} are missing."
            )

        arrays = {b: bands[b] for b in self.required_bands}
        values, valid = self._compute(**arrays)

        return IndexResult(
            name=self.name,
            values=values,
            valid_mask=valid,
            description=self.description,
            positive_interpretation=self.positive_interpretation,
            negative_interpretation=self.negative_interpretation,
        )

    def compute_from_stack(
        self,
        data: np.ndarray,
        band_names: List[str],
    ) -> IndexResult:
        """
        Compute the index from a multi-band array and band name list.

        Args:
            data:       Array of shape [n_bands, H, W], float32 reflectance.
            band_names: List of band names matching the first axis.

        Returns:
            IndexResult.
        """
        bands = {name: data[i] for i, name in enumerate(band_names)}
        return self.compute_from_dict(bands)

    @abstractmethod
    def _compute(self, **band_arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the index formula.

        Returns:
            Tuple of (values [H, W], valid_mask [H, W]).
        """
        ...


def _normalized_difference(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute (A - B) / (A + B) with NaN handling and zero-division protection.

    Args:
        a: First band (float32).
        b: Second band (float32).

    Returns:
        Tuple of (index_values, valid_mask).
        Index values are clipped to [-1, 1].
        Valid mask is True where both inputs are finite.
    """
    valid = np.isfinite(a) & np.isfinite(b)
    denominator = a + b + EPSILON  # Avoid exact zero division
    values = np.where(valid, (a - b) / denominator, np.nan)
    values = np.clip(values, -1.0, 1.0)
    return values.astype(np.float32), valid


class NDVIIndex(SpectralIndex):
    """
    Normalized Difference Vegetation Index.

    Formula: NDVI = (NIR - Red) / (NIR + Red)
    Bands:   B08 (NIR, 842 nm), B04 (Red, 665 nm)
    Range:   [-1, 1]

    Interpretation:
      NDVI > 0.5: Dense healthy vegetation
      0.2–0.5:    Sparse/moderate vegetation
      0.0–0.2:    Bare soil, rock, urban
      < 0.0:      Water, snow, cloud

    Scientific caution:
      NDVI is sensitive to illumination, atmospheric conditions, and
      soil background. Seasonal variation can cause NDVI to change
      substantially without any anthropogenic activity.
      Always compare observations from the same season where possible.
    """

    name = "NDVI"
    description = "Normalized Difference Vegetation Index"
    required_bands = [BAND_NIR, BAND_RED]
    positive_interpretation = "Possible dense vegetation (NDVI > 0.5)"
    negative_interpretation = "Possible water or snow (NDVI < 0)"

    def _compute(self, B08: np.ndarray, B04: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _normalized_difference(B08, B04)


class NDWIIndex(SpectralIndex):
    """
    Normalized Difference Water Index.

    Formula: NDWI = (Green - NIR) / (Green + NIR)
    Bands:   B03 (Green, 560 nm), B08 (NIR, 842 nm)
    Range:   [-1, 1]

    Interpretation:
      NDWI > 0:   Open water, flooded areas
      NDWI < 0:   Vegetation, soil, built-up

    Scientific caution:
      NDWI can be confused by bright surfaces (snow, sand) and
      thin cloud. Use in conjunction with cloud masking.
    """

    name = "NDWI"
    description = "Normalized Difference Water Index"
    required_bands = [BAND_GREEN, BAND_NIR]
    positive_interpretation = "Possible open water or wetland (NDWI > 0)"
    negative_interpretation = "Possible vegetation or dry land (NDWI < 0)"

    def _compute(self, B03: np.ndarray, B08: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _normalized_difference(B03, B08)


class NDBIIndex(SpectralIndex):
    """
    Normalized Difference Built-up Index.

    Formula: NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Bands:   B11 (SWIR1, 1610 nm), B08 (NIR, 842 nm)
    Range:   [-1, 1]

    Interpretation:
      NDBI > 0:   Built-up areas, bare soil, urban
      NDBI < 0:   Vegetation, water

    Scientific caution:
      NDBI alone cannot reliably distinguish built-up from bare soil.
      Use alongside NDVI — low NDVI + high NDBI increases confidence
      of built-up/urban surface.
      B11 has 20 m native resolution; resampling to 10 m introduces
      additional spatial uncertainty.
    """

    name = "NDBI"
    description = "Normalized Difference Built-up Index"
    required_bands = [BAND_SWIR1, BAND_NIR]
    positive_interpretation = "Possible built-up or bare soil (NDBI > 0)"
    negative_interpretation = "Possible vegetation or water (NDBI < 0)"

    def _compute(self, B11: np.ndarray, B08: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _normalized_difference(B11, B08)


# ── Index Registry ────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, SpectralIndex] = {}


def register_index(index: SpectralIndex) -> None:
    """Register a SpectralIndex instance in the global registry."""
    _REGISTRY[index.name] = index


def get_index(name: str) -> SpectralIndex:
    """
    Return a registered SpectralIndex by name.

    Args:
        name: Index name (e.g. 'NDVI').

    Raises:
        KeyError: If the index is not registered.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown spectral index '{name}'. "
            f"Available: {available}"
        )
    return _REGISTRY[name]


def list_indices() -> List[str]:
    """Return names of all registered spectral indices."""
    return list(_REGISTRY.keys())


def compute_all_indices(
    data: np.ndarray,
    band_names: List[str],
) -> Dict[str, IndexResult]:
    """
    Compute all registered spectral indices for a given image.

    Only computes indices for which all required bands are available.

    Args:
        data:       Array of shape [n_bands, H, W] (float32 reflectance).
        band_names: List of band names matching the first axis.

    Returns:
        Dict mapping index name → IndexResult.
    """
    available_bands = set(band_names)
    results = {}

    for name, index in _REGISTRY.items():
        if all(b in available_bands for b in index.required_bands):
            try:
                results[name] = index.compute_from_stack(data, band_names)
                logger.debug("Computed %s.", name)
            except Exception as exc:
                logger.warning("Failed to compute %s: %s", name, exc)
        else:
            missing = [b for b in index.required_bands if b not in available_bands]
            logger.debug(
                "Skipping %s: missing bands %s", name, missing
            )

    return results


def compute_index_change(
    before_result: IndexResult,
    after_result: IndexResult,
) -> IndexResult:
    """
    Compute the change in a spectral index between two observations.

    Change = after - before

    Both IndexResults must have the same spatial dimensions.
    The valid mask is the AND of both observation masks.

    Args:
        before_result: Index computed on the 'before' image.
        after_result:  Index computed on the 'after' image.

    Returns:
        IndexResult with difference values.
        Positive = increase, Negative = decrease.
    """
    if before_result.values.shape != after_result.values.shape:
        raise ValueError(
            f"Shape mismatch: {before_result.values.shape} vs {after_result.values.shape}"
        )

    valid = before_result.valid_mask & after_result.valid_mask
    diff = np.where(valid, after_result.values - before_result.values, np.nan)

    return IndexResult(
        name=f"{before_result.name}_change",
        values=diff.astype(np.float32),
        valid_mask=valid,
        description=f"Change in {before_result.name} (after − before)",
        positive_interpretation=f"Possible increase in {before_result.name}",
        negative_interpretation=f"Possible decrease in {before_result.name}",
    )


# ── Register built-in indices ──────────────────────────────────────────────

register_index(NDVIIndex())
register_index(NDWIIndex())
register_index(NDBIIndex())
