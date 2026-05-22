"""Sentinel-2 optical preprocessing: cloud masking, band selection, normalisation."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from typing import Tuple

# Scene Classification Layer values that represent cloud / shadow / invalid
_SCL_VALID = {4, 5, 6}   # Vegetation, Not-Vegetated, Water
_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]


class Sentinel2Preprocessor:
    """
    Loads Sentinel-2 Level-2A (BOA) imagery.  Bands are resampled to a
    common 10 m resolution, clouds / shadows are masked via the SCL band,
    and values are scaled to [0, 1] reflectance.
    """

    def __init__(
        self,
        bands: list[str] = _BAND_ORDER,
        cloud_prob_threshold: int = 20,
        scale_factor: float = 10_000.0,
        target_resolution: int = 10,
    ) -> None:
        self.bands = list(bands)
        self.cloud_prob_threshold = cloud_prob_threshold
        self.scale_factor = scale_factor
        self.target_resolution = target_resolution

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(
        self, scene_path: str | Path
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Parameters
        ----------
        scene_path : path to a Sentinel-2 SAFE directory or multi-band GeoTIFF

        Returns
        -------
        array : np.ndarray  shape (C, H, W) float32, values in [0, 1]
        mask  : np.ndarray  shape (H, W) bool, True = valid pixel
        meta  : dict        rasterio profile
        """
        scene_path = Path(scene_path)
        bands, scl, meta = self._load_bands(scene_path)
        valid_mask = self._build_cloud_mask(scl)
        bands = self._scale(bands)
        bands = self._apply_mask(bands, valid_mask)
        return bands.astype(np.float32), valid_mask, meta

    def process_and_save(
        self, scene_path: str | Path, output_path: str | Path
    ) -> None:
        array, mask, meta = self.process(scene_path)
        meta.update(count=array.shape[0], dtype="float32", nodata=0.0)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(array)

    def compute_indices(self, bands: np.ndarray) -> np.ndarray:
        """
        Append NDVI and NDWI channels to the band stack.

        Expects bands ordered as B2 B3 B4 B8 B11 B12 (indices 0-5).
        Returns array of shape (C+2, H, W).
        """
        b3  = bands[1]  # Green
        b4  = bands[2]  # Red
        b8  = bands[3]  # NIR
        b11 = bands[4]  # SWIR1

        ndvi = _safe_ratio(b8 - b4, b8 + b4)
        ndwi = _safe_ratio(b3 - b8, b3 + b8)
        return np.concatenate([bands, ndvi[None], ndwi[None]], axis=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_bands(
        self, scene_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        if scene_path.is_file():
            return self._load_multiband_tif(scene_path)
        return self._load_safe_directory(scene_path)

    def _load_multiband_tif(self, path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float64)
            meta = src.profile.copy()
        # Last channel is treated as SCL when loading a single file
        scl = data[-1].astype(np.uint8)
        return data[:-1], scl, meta

    def _load_safe_directory(
        self, directory: Path
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        bands_data = []
        meta = None
        reference_shape = None

        for band_name in self.bands:
            path = self._find_band(directory, band_name)
            arr, band_meta, shape = self._read_resample(path, reference_shape)
            if reference_shape is None:
                reference_shape = shape
                meta = band_meta
            bands_data.append(arr)

        scl_path = self._find_band(directory, "SCL")
        scl, _, _ = self._read_resample(scl_path, reference_shape)

        return np.stack(bands_data, axis=0), scl.astype(np.uint8), meta

    def _find_band(self, directory: Path, band_name: str) -> Path:
        for pattern in [f"*_{band_name}_*.jp2", f"*_{band_name}_*.tif"]:
            hits = list(directory.rglob(pattern))
            if hits:
                # Prefer 10 m resolution for B11/B12 if available
                return hits[0]
        raise FileNotFoundError(f"Band {band_name} not found in {directory}")

    def _read_resample(
        self, path: Path, target_shape: tuple | None
    ) -> Tuple[np.ndarray, dict, tuple]:
        with rasterio.open(path) as src:
            if target_shape is None:
                data = src.read(1).astype(np.float64)
                return data, src.profile.copy(), data.shape
            data = src.read(
                1,
                out_shape=(1, *target_shape),
                resampling=Resampling.bilinear,
            )[0].astype(np.float64)
            return data, src.profile.copy(), target_shape

    def _build_cloud_mask(self, scl: np.ndarray) -> np.ndarray:
        """Return bool array True = valid (non-cloudy, non-shadow) pixel."""
        valid = np.zeros(scl.shape, dtype=bool)
        for v in _SCL_VALID:
            valid |= scl == v
        return valid

    def _scale(self, bands: np.ndarray) -> np.ndarray:
        return np.clip(bands / self.scale_factor, 0.0, 1.0)

    def _apply_mask(self, bands: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = bands.copy()
        out[:, ~mask] = 0.0
        return out


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(np.abs(den) > 1e-8, num / den, 0.0)
