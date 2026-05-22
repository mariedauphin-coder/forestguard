"""Sentinel-1 SAR preprocessing: speckle filtering, dB conversion, normalisation."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from typing import Tuple


class Sentinel1Preprocessor:
    """
    Loads raw Sentinel-1 GRD imagery (VV/VH polarisations), applies a Lee
    speckle filter, converts linear power to dB, clips outliers, and
    normalises to [0, 1] for model ingestion.
    """

    def __init__(
        self,
        polarizations: list[str] = ("VV", "VH"),
        filter_size: int = 5,
        clip_min: float = -25.0,
        clip_max: float = 0.0,
        target_resolution: int = 10,
    ) -> None:
        self.polarizations = list(polarizations)
        self.filter_size = filter_size
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.target_resolution = target_resolution

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(self, scene_path: str | Path) -> Tuple[np.ndarray, dict]:
        """
        Process a single Sentinel-1 scene directory or multi-band GeoTIFF.

        Returns
        -------
        array : np.ndarray  shape (C, H, W) float32, values in [0, 1]
        meta  : dict        rasterio profile of the output raster
        """
        scene_path = Path(scene_path)
        bands, meta = self._load_bands(scene_path)
        bands = self._apply_lee_filter(bands)
        bands = self._to_db(bands)
        bands = self._clip_and_normalise(bands)
        return bands.astype(np.float32), meta

    def process_and_save(
        self, scene_path: str | Path, output_path: str | Path
    ) -> None:
        array, meta = self.process(scene_path)
        meta.update(count=array.shape[0], dtype="float32")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(array)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_bands(self, scene_path: Path) -> Tuple[np.ndarray, dict]:
        """Load VV and VH bands from a multi-band GeoTIFF or directory."""
        if scene_path.is_file() and scene_path.suffix in (".tif", ".tiff"):
            return self._load_multiband_tif(scene_path)
        return self._load_directory(scene_path)

    def _load_multiband_tif(self, path: Path) -> Tuple[np.ndarray, dict]:
        with rasterio.open(path) as src:
            data = src.read().astype(np.float64)
            meta = src.profile.copy()
        return data, meta

    def _load_directory(self, directory: Path) -> Tuple[np.ndarray, dict]:
        bands = []
        meta = None
        for pol in self.polarizations:
            candidates = list(directory.glob(f"*{pol}*.tif")) + list(
                directory.glob(f"*{pol.lower()}*.tif")
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No file found for polarisation {pol} in {directory}"
                )
            with rasterio.open(candidates[0]) as src:
                if meta is None:
                    meta = src.profile.copy()
                bands.append(src.read(1).astype(np.float64))
        return np.stack(bands, axis=0), meta

    def _apply_lee_filter(self, bands: np.ndarray) -> np.ndarray:
        """
        Per-channel Lee speckle filter (simplified adaptive version).
        Assumes linear power units.
        """
        from scipy.ndimage import uniform_filter

        out = np.empty_like(bands)
        for i, band in enumerate(bands):
            mean = uniform_filter(band, self.filter_size)
            mean_sq = uniform_filter(band ** 2, self.filter_size)
            variance = mean_sq - mean ** 2

            # ENL (equivalent number of looks) estimate ≈ 4.9 for IW GRD
            noise_var = np.mean(variance) / (np.mean(mean) ** 2 + 1e-10)
            weight = np.where(
                variance > 0,
                variance / (variance + noise_var * mean ** 2 + 1e-10),
                0.0,
            )
            out[i] = mean + weight * (band - mean)
        return out

    def _to_db(self, bands: np.ndarray) -> np.ndarray:
        """Convert linear power to dB: 10 * log10(power)."""
        return 10.0 * np.log10(np.clip(bands, 1e-10, None))

    def _clip_and_normalise(self, bands: np.ndarray) -> np.ndarray:
        clipped = np.clip(bands, self.clip_min, self.clip_max)
        return (clipped - self.clip_min) / (self.clip_max - self.clip_min)
