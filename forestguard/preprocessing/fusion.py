"""Pixel-level fusion of Sentinel-1 SAR and Sentinel-2 optical data + patch extraction."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from typing import Iterator, Tuple

from .sentinel1 import Sentinel1Preprocessor
from .sentinel2 import Sentinel2Preprocessor


class FusionPipeline:
    """
    Aligns SAR and optical rasters to a common grid, stacks channels, then
    tiles the fused stack into fixed-size patches for training or inference.

    Output channel order (8 channels by default):
        0: S1 VV
        1: S1 VH
        2: S2 B2 (Blue)
        3: S2 B3 (Green)
        4: S2 B4 (Red)
        5: S2 B8 (NIR)
        6: S2 B11 (SWIR1)
        7: S2 B12 (SWIR2)
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 128,
        sar_preprocessor: Sentinel1Preprocessor | None = None,
        optical_preprocessor: Sentinel2Preprocessor | None = None,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.sar = sar_preprocessor or Sentinel1Preprocessor()
        self.optical = optical_preprocessor or Sentinel2Preprocessor()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fuse(
        self,
        sar_path: str | Path,
        optical_path: str | Path,
        output_path: str | Path | None = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Fuse a SAR and optical scene into a single multi-channel raster.

        Returns
        -------
        fused : np.ndarray  shape (C, H, W) float32
        meta  : dict        rasterio profile (CRS, transform, …)
        """
        sar_arr, sar_meta = self.sar.process(sar_path)
        opt_arr, opt_mask, opt_meta = self.optical.process(optical_path)

        # Reproject SAR to optical grid (optical is the reference)
        sar_aligned = self._align_to_reference(sar_arr, sar_meta, opt_meta)

        fused = np.concatenate([sar_aligned, opt_arr], axis=0)

        if output_path is not None:
            meta = opt_meta.copy()
            meta.update(count=fused.shape[0], dtype="float32")
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(fused)

        return fused.astype(np.float32), opt_meta

    def extract_patches(
        self,
        fused_array: np.ndarray,
        label_array: np.ndarray | None = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray | None, Tuple[int, int]]]:
        """
        Yield (patch, label_patch, (row, col)) tuples with the given stride.

        label_array : optional (H, W) binary mask — yielded alongside each patch
        """
        _, H, W = fused_array.shape
        p, s = self.patch_size, self.stride

        for row in range(0, H - p + 1, s):
            for col in range(0, W - p + 1, s):
                patch = fused_array[:, row : row + p, col : col + p]
                label_patch = (
                    label_array[row : row + p, col : col + p]
                    if label_array is not None
                    else None
                )
                yield patch, label_patch, (row, col)

    def fuse_pair(
        self,
        before_sar: str | Path,
        before_optical: str | Path,
        after_sar: str | Path,
        after_optical: str | Path,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Fuse two temporal scenes (before / after) for change detection.

        Returns
        -------
        before : np.ndarray  (C, H, W)
        after  : np.ndarray  (C, H, W)
        meta   : dict
        """
        before, meta = self.fuse(before_sar, before_optical)
        after, _ = self.fuse(after_sar, after_optical)

        # Ensure identical spatial extent
        h = min(before.shape[1], after.shape[1])
        w = min(before.shape[2], after.shape[2])
        return before[:, :h, :w], after[:, :h, :w], meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _align_to_reference(
        self,
        source_array: np.ndarray,
        source_meta: dict,
        ref_meta: dict,
    ) -> np.ndarray:
        """Reproject source array into reference grid using bilinear resampling."""
        C = source_array.shape[0]
        dst_height = ref_meta["height"]
        dst_width = ref_meta["width"]
        aligned = np.zeros((C, dst_height, dst_width), dtype=np.float32)

        for c in range(C):
            reproject(
                source=source_array[c],
                destination=aligned[c],
                src_transform=source_meta["transform"],
                src_crs=source_meta["crs"],
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
            )
        return aligned
