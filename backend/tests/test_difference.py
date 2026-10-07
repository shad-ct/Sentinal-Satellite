"""Unit tests for classical change detection (difference, thresholding, morphology)."""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.analysis.difference import (
    compute_difference,
    compute_change_score,
    apply_threshold,
    apply_morphological_cleanup,
    detect_changes,
)


class TestComputeDifference:
    def test_identical_images_zero_difference(self):
        img = np.random.rand(3, 50, 50).astype(np.float32)
        diff = compute_difference(img, img)
        assert np.all(diff == 0.0), "Identical images should produce zero difference"

    def test_shape_preserved(self):
        before = np.random.rand(4, 32, 32).astype(np.float32)
        after = np.random.rand(4, 32, 32).astype(np.float32)
        diff = compute_difference(before, after)
        assert diff.shape == before.shape

    def test_shape_mismatch_raises(self):
        before = np.zeros((3, 10, 10))
        after = np.zeros((3, 20, 20))
        with pytest.raises(ValueError):
            compute_difference(before, after)

    def test_output_range(self):
        before = np.random.rand(3, 50, 50).astype(np.float32)
        after = np.random.rand(3, 50, 50).astype(np.float32)
        diff = compute_difference(before, after)
        assert diff.min() >= 0.0
        assert diff.max() <= 1.0

    def test_valid_mask_zeros_invalid(self):
        before = np.ones((2, 10, 10), dtype=np.float32)
        after = np.zeros((2, 10, 10), dtype=np.float32)
        valid = np.zeros((10, 10), dtype=bool)
        valid[5:, 5:] = True
        diff = compute_difference(before, after, valid_mask=valid)
        assert np.all(diff[:, :5, :] == 0.0), "Invalid pixels should be zeroed"


class TestComputeChangeScore:
    def test_single_band(self):
        diff = np.full((1, 20, 20), 0.5, dtype=np.float32)
        score = compute_change_score(diff)
        assert score.shape == (20, 20)
        assert np.allclose(score, 0.5)

    def test_mean_across_bands(self):
        diff = np.array([
            np.full((10, 10), 0.2),
            np.full((10, 10), 0.8),
        ], dtype=np.float32)
        score = compute_change_score(diff)
        assert np.allclose(score, 0.5)


class TestApplyThreshold:
    def test_threshold_produces_binary(self):
        score = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        mask = apply_threshold(score, threshold=0.5)
        assert set(np.unique(mask)).issubset({0, 1})

    def test_above_threshold_is_one(self):
        score = np.full((5, 5), 0.8, dtype=np.float32)
        mask = apply_threshold(score, threshold=0.5)
        assert np.all(mask == 1)

    def test_below_threshold_is_zero(self):
        score = np.full((5, 5), 0.2, dtype=np.float32)
        mask = apply_threshold(score, threshold=0.5)
        assert np.all(mask == 0)

    def test_invalid_threshold_raises(self):
        score = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            apply_threshold(score, threshold=1.5)


class TestMorphologicalCleanup:
    def test_removes_small_noise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1  # Single pixel — should be removed
        cleaned = apply_morphological_cleanup(mask, min_region_pixels=10)
        assert cleaned[50, 50] == 0, "Single pixel should be removed by morphological cleanup"

    def test_keeps_large_region(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 1  # Large region (40×40 = 1600 pixels)
        cleaned = apply_morphological_cleanup(mask, min_region_pixels=10)
        assert cleaned[50, 50] == 1, "Large region should be preserved"


class TestDetectChanges:
    def test_no_change_low_score(self):
        img = np.random.rand(3, 50, 50).astype(np.float32) * 0.01
        result = detect_changes(img, img)
        assert result.change_percentage < 1.0, "Identical images should show <1% change"

    def test_full_change_high_score(self):
        before = np.zeros((3, 50, 50), dtype=np.float32)
        after = np.ones((3, 50, 50), dtype=np.float32)
        result = detect_changes(before, after, threshold=0.1)
        assert result.change_percentage > 50.0, "Completely different images should show high change"

    def test_result_has_all_fields(self):
        img = np.random.rand(2, 30, 30).astype(np.float32)
        result = detect_changes(img, img * 1.5, threshold=0.1)
        assert hasattr(result, "raw_difference")
        assert hasattr(result, "change_score")
        assert hasattr(result, "binary_mask")
        assert hasattr(result, "valid_mask")
        assert hasattr(result, "n_changed_pixels")
        assert hasattr(result, "n_valid_pixels")
