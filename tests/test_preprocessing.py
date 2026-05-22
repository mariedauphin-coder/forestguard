"""Unit tests for preprocessing modules (no real rasters required)."""

import numpy as np
import pytest
from forestguard.preprocessing.sentinel1 import Sentinel1Preprocessor
from forestguard.preprocessing.sentinel2 import Sentinel2Preprocessor, _safe_ratio
from forestguard.preprocessing.fusion import FusionPipeline


class TestSentinel1Preprocessor:
    def setup_method(self):
        self.prep = Sentinel1Preprocessor(clip_min=-25.0, clip_max=0.0)

    def test_clip_and_normalise_range(self):
        bands = np.array([[[-30.0, -12.5, 0.0, 5.0]]])
        result = self.prep._clip_and_normalise(bands)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_to_db_converts_linear(self):
        linear = np.array([[[1.0, 10.0, 0.01]]])
        db = self.prep._to_db(linear)
        np.testing.assert_allclose(db[0, 0, 0], 0.0,  atol=1e-5)
        np.testing.assert_allclose(db[0, 0, 1], 10.0, atol=1e-5)

    def test_lee_filter_output_shape(self):
        bands = np.random.rand(2, 64, 64)
        out = self.prep._apply_lee_filter(bands)
        assert out.shape == bands.shape


class TestSentinel2Preprocessor:
    def test_scale_clips_to_unit_range(self):
        prep = Sentinel2Preprocessor()
        bands = np.array([[[0, 5000, 10000, 15000]]], dtype=float)
        scaled = prep._scale(bands)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_cloud_mask_valid_pixels(self):
        prep = Sentinel2Preprocessor()
        scl = np.array([[4, 5, 6, 3, 8, 9]], dtype=np.uint8)
        mask = prep._build_cloud_mask(scl)
        assert mask[0, 0] and mask[0, 1] and mask[0, 2]
        assert not mask[0, 3] and not mask[0, 4]

    def test_safe_ratio_no_divide_by_zero(self):
        num = np.array([1.0, 0.0])
        den = np.array([0.0, 0.0])
        result = _safe_ratio(num, den)
        assert not np.any(np.isinf(result))
        assert not np.any(np.isnan(result))


class TestFusionPipeline:
    def test_extract_patches_count(self):
        pipeline = FusionPipeline(patch_size=32, stride=16)
        fused = np.random.rand(8, 64, 64).astype(np.float32)
        patches = list(pipeline.extract_patches(fused))
        assert len(patches) > 0
        for patch, label, (r, c) in patches:
            assert patch.shape == (8, 32, 32)

    def test_extract_patches_with_label(self):
        pipeline = FusionPipeline(patch_size=32, stride=32)
        fused = np.random.rand(8, 64, 64).astype(np.float32)
        label = np.zeros((64, 64), dtype=np.uint8)
        for _, label_patch, _ in pipeline.extract_patches(fused, label):
            assert label_patch is not None
            assert label_patch.shape == (32, 32)
