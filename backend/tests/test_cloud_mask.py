"""Unit tests for cloud masking (SCL-based)."""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.preprocessing.cloud_mask import (
    SCLClass,
    MaskClass,
    create_cloud_mask,
    apply_cloud_mask,
    calculate_valid_pixel_percentage,
    extract_scl_from_multiband,
)


class TestCreateCloudMask:
    def _scl(self, *values):
        return np.array([[v] for v in values], dtype=np.uint8)

    def test_valid_pixels(self):
        scl = np.full((5, 5), SCLClass.VEGETATION, dtype=np.uint8)
        mask = create_cloud_mask(scl)
        assert np.all(mask == MaskClass.VALID)

    def test_cloud_high_probability(self):
        scl = np.full((5, 5), SCLClass.CLOUD_HIGH, dtype=np.uint8)
        mask = create_cloud_mask(scl)
        assert np.all(mask == MaskClass.CLOUD)

    def test_cloud_medium_probability(self):
        scl = np.full((5, 5), SCLClass.CLOUD_MEDIUM, dtype=np.uint8)
        mask = create_cloud_mask(scl)
        assert np.all(mask == MaskClass.CLOUD)

    def test_thin_cirrus_default(self):
        scl = np.full((5, 5), SCLClass.THIN_CIRRUS, dtype=np.uint8)
        mask = create_cloud_mask(scl, include_cirrus=True)
        assert np.all(mask == MaskClass.CLOUD)

    def test_thin_cirrus_excluded(self):
        scl = np.full((5, 5), SCLClass.THIN_CIRRUS, dtype=np.uint8)
        mask = create_cloud_mask(scl, include_cirrus=False)
        assert np.all(mask == MaskClass.VALID)

    def test_cloud_shadow(self):
        scl = np.full((5, 5), SCLClass.CLOUD_SHADOW, dtype=np.uint8)
        mask = create_cloud_mask(scl, include_shadow=True)
        assert np.all(mask == MaskClass.CLOUD_SHADOW)

    def test_shadow_excluded(self):
        scl = np.full((5, 5), SCLClass.CLOUD_SHADOW, dtype=np.uint8)
        mask = create_cloud_mask(scl, include_shadow=False)
        assert np.all(mask == MaskClass.VALID)

    def test_no_data(self):
        scl = np.full((5, 5), SCLClass.NO_DATA, dtype=np.uint8)
        mask = create_cloud_mask(scl)
        assert np.all(mask == MaskClass.INVALID)

    def test_mixed_scene(self):
        scl = np.array([
            [SCLClass.VEGETATION, SCLClass.CLOUD_HIGH],
            [SCLClass.WATER, SCLClass.CLOUD_SHADOW],
        ], dtype=np.uint8)
        mask = create_cloud_mask(scl)
        assert mask[0, 0] == MaskClass.VALID
        assert mask[0, 1] == MaskClass.CLOUD
        assert mask[1, 0] == MaskClass.VALID
        assert mask[1, 1] == MaskClass.CLOUD_SHADOW


class TestApplyCloudMask:
    def test_masked_pixels_become_nan(self):
        data = np.ones((3, 4, 4), dtype=np.float32)
        cloud_mask = np.zeros((4, 4), dtype=np.uint8)
        cloud_mask[0, 0] = MaskClass.CLOUD
        result = apply_cloud_mask(data, cloud_mask)
        assert np.isnan(result[:, 0, 0]).all()
        assert not np.isnan(result[:, 1, 1]).any()

    def test_custom_fill_value(self):
        data = np.ones((2, 3, 3), dtype=np.float32)
        cloud_mask = np.full((3, 3), MaskClass.CLOUD, dtype=np.uint8)
        result = apply_cloud_mask(data, cloud_mask, fill_value=-9999.0)
        assert np.all(result == -9999.0)


class TestValidPixelPercentage:
    def test_all_valid(self):
        mask = np.zeros((10, 10), dtype=np.uint8)  # All VALID
        stats = calculate_valid_pixel_percentage(mask)
        assert stats["valid_pct"] == pytest.approx(100.0)

    def test_all_cloud(self):
        mask = np.full((10, 10), MaskClass.CLOUD, dtype=np.uint8)
        stats = calculate_valid_pixel_percentage(mask)
        assert stats["cloud_pct"] == pytest.approx(100.0)
        assert stats["valid_pct"] == pytest.approx(0.0)

    def test_half_cloud(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = MaskClass.CLOUD
        stats = calculate_valid_pixel_percentage(mask)
        assert abs(stats["cloud_pct"] - 50.0) < 0.5
        assert abs(stats["valid_pct"] - 50.0) < 0.5


class TestExtractSCL:
    def test_scl_found(self):
        data = np.random.randint(0, 12, (4, 10, 10), dtype=np.uint8).astype(np.float32)
        bands = ["B04", "B08", "B11", "SCL"]
        scl = extract_scl_from_multiband(data, bands)
        assert scl is not None
        assert scl.shape == (10, 10)
        assert np.array_equal(scl, data[3].astype(np.uint8))

    def test_scl_not_found(self):
        data = np.random.rand(3, 10, 10).astype(np.float32)
        bands = ["B04", "B08", "B11"]
        scl = extract_scl_from_multiband(data, bands)
        assert scl is None
