"""
Unit tests for spectral index calculations.

Tests NDVI, NDWI, NDBI formulas with known inputs,
edge cases (zero denominator, NaN inputs, all-cloud masks).
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.analysis.indices import (
    NDVIIndex,
    NDWIIndex,
    NDBIIndex,
    compute_index_change,
    compute_all_indices,
    list_indices,
    get_index,
    EPSILON,
)


class TestNDVI:
    """Tests for the Normalized Difference Vegetation Index."""

    def test_basic_formula(self):
        """NDVI = (NIR - Red) / (NIR + Red)"""
        nir = np.array([[0.6, 0.4]], dtype=np.float32)
        red = np.array([[0.1, 0.1]], dtype=np.float32)
        expected_0 = (0.6 - 0.1) / (0.6 + 0.1 + EPSILON)
        expected_1 = (0.4 - 0.1) / (0.4 + 0.1 + EPSILON)

        idx = NDVIIndex()
        result = idx.compute_from_dict({"B08": nir, "B04": red})

        assert result.name == "NDVI"
        assert result.values.shape == (1, 2)
        assert abs(result.values[0, 0] - expected_0) < 1e-4
        assert abs(result.values[0, 1] - expected_1) < 1e-4

    def test_water_body(self):
        """Water bodies have strongly negative NDVI."""
        nir = np.array([[0.03]], dtype=np.float32)
        red = np.array([[0.08]], dtype=np.float32)
        idx = NDVIIndex()
        result = idx.compute_from_dict({"B08": nir, "B04": red})
        assert result.values[0, 0] < 0, "Water should produce negative NDVI"

    def test_dense_vegetation(self):
        """Dense vegetation has NDVI close to 1."""
        nir = np.array([[0.85]], dtype=np.float32)
        red = np.array([[0.05]], dtype=np.float32)
        idx = NDVIIndex()
        result = idx.compute_from_dict({"B08": nir, "B04": red})
        assert result.values[0, 0] > 0.5, "Dense vegetation should have NDVI > 0.5"

    def test_zero_denominator(self):
        """When NIR = Red = 0, NDVI should be ~0 (not NaN/inf)."""
        nir = np.array([[0.0]], dtype=np.float32)
        red = np.array([[0.0]], dtype=np.float32)
        idx = NDVIIndex()
        result = idx.compute_from_dict({"B08": nir, "B04": red})
        assert np.isfinite(result.values[0, 0]), "Zero denominator must not produce inf/nan"

    def test_nan_input_propagates(self):
        """NaN inputs produce NaN outputs and False valid mask."""
        nir = np.array([[np.nan, 0.5]], dtype=np.float32)
        red = np.array([[0.1, 0.1]], dtype=np.float32)
        idx = NDVIIndex()
        result = idx.compute_from_dict({"B08": nir, "B04": red})
        assert not result.valid_mask[0, 0], "NaN pixel should be invalid"
        assert result.valid_mask[0, 1], "Valid pixel should be valid"

    def test_clipped_range(self):
        """NDVI output is always in [-1, 1]."""
        nir = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
        red = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
        idx = NDVIIndex()
        result = idx.compute_from_dict({"B08": nir, "B04": red})
        assert result.values.min() >= -1.0
        assert result.values.max() <= 1.0

    def test_missing_band_raises(self):
        """Missing required band raises KeyError."""
        idx = NDVIIndex()
        with pytest.raises(KeyError, match="B08"):
            idx.compute_from_dict({"B04": np.zeros((5, 5))})


class TestNDWI:
    """Tests for the Normalized Difference Water Index."""

    def test_water_positive(self):
        """Open water: NDWI > 0."""
        green = np.array([[0.25]], dtype=np.float32)
        nir = np.array([[0.05]], dtype=np.float32)
        idx = NDWIIndex()
        result = idx.compute_from_dict({"B03": green, "B08": nir})
        assert result.values[0, 0] > 0, "Open water should have positive NDWI"

    def test_vegetation_negative(self):
        """Dense vegetation: NDWI < 0."""
        green = np.array([[0.06]], dtype=np.float32)
        nir = np.array([[0.80]], dtype=np.float32)
        idx = NDWIIndex()
        result = idx.compute_from_dict({"B03": green, "B08": nir})
        assert result.values[0, 0] < 0, "Vegetation should have negative NDWI"

    def test_formula_correct(self):
        """NDWI = (Green - NIR) / (Green + NIR)."""
        green = np.array([[0.3]], dtype=np.float32)
        nir = np.array([[0.2]], dtype=np.float32)
        expected = (0.3 - 0.2) / (0.3 + 0.2 + EPSILON)
        idx = NDWIIndex()
        result = idx.compute_from_dict({"B03": green, "B08": nir})
        assert abs(result.values[0, 0] - expected) < 1e-4


class TestNDBI:
    """Tests for the Normalized Difference Built-up Index."""

    def test_formula_correct(self):
        """NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)."""
        swir = np.array([[0.4]], dtype=np.float32)
        nir = np.array([[0.2]], dtype=np.float32)
        expected = (0.4 - 0.2) / (0.4 + 0.2 + EPSILON)
        idx = NDBIIndex()
        result = idx.compute_from_dict({"B11": swir, "B08": nir})
        assert abs(result.values[0, 0] - expected) < 1e-4


class TestIndexChange:
    """Tests for compute_index_change."""

    def test_positive_change(self):
        """After > before → positive change."""
        from backend.app.analysis.indices import IndexResult
        before = IndexResult("NDVI", np.array([[0.2, 0.3]]), np.ones((1, 2), dtype=bool))
        after = IndexResult("NDVI", np.array([[0.5, 0.3]]), np.ones((1, 2), dtype=bool))
        change = compute_index_change(before, after)
        assert abs(change.values[0, 0] - 0.3) < 1e-4
        assert abs(change.values[0, 1]) < 1e-4

    def test_shape_mismatch_raises(self):
        from backend.app.analysis.indices import IndexResult
        a = IndexResult("NDVI", np.zeros((2, 3)), np.ones((2, 3), dtype=bool))
        b = IndexResult("NDVI", np.zeros((3, 2)), np.ones((3, 2), dtype=bool))
        with pytest.raises(ValueError):
            compute_index_change(a, b)


class TestRegistry:
    """Tests for the index registry."""

    def test_all_built_in_indices_registered(self):
        assert "NDVI" in list_indices()
        assert "NDWI" in list_indices()
        assert "NDBI" in list_indices()

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_index("FAKEINDEX_XYZ")

    def test_compute_all_with_sufficient_bands(self):
        """compute_all_indices runs without error when all bands are present."""
        data = np.random.uniform(0.05, 0.5, (6, 50, 50)).astype(np.float32)
        bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
        results = compute_all_indices(data, bands)
        assert "NDVI" in results
        assert "NDWI" in results
        assert "NDBI" in results

    def test_compute_all_missing_swir_skips_ndbi(self):
        """NDBI is skipped when B11 is not present."""
        data = np.random.uniform(0.05, 0.5, (4, 50, 50)).astype(np.float32)
        bands = ["B02", "B03", "B04", "B08"]
        results = compute_all_indices(data, bands)
        assert "NDVI" in results
        assert "NDWI" in results
        assert "NDBI" not in results
