"""Morphological filtering to remove noise from raw model prediction masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, label


class MorphologicalFilter:
    """
    Applies morphological opening (remove speckle) and closing (fill holes)
    to a binary deforestation mask, then removes connected components
    smaller than min_area_pixels.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        min_area_pixels: int = 100,
    ) -> None:
        self.kernel_size = kernel_size
        self.min_area_pixels = min_area_pixels
        self._structure = np.ones((kernel_size, kernel_size), dtype=bool)

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        mask : (H, W) bool or uint8 binary array

        Returns
        -------
        filtered : (H, W) uint8
        """
        binary = mask.astype(bool)
        binary = binary_opening(binary, structure=self._structure)
        binary = binary_closing(binary, structure=self._structure)
        return self._remove_small_components(binary).astype(np.uint8)

    def _remove_small_components(self, binary: np.ndarray) -> np.ndarray:
        labeled, n_components = label(binary)
        out = np.zeros_like(binary)
        for comp_id in range(1, n_components + 1):
            component = labeled == comp_id
            if component.sum() >= self.min_area_pixels:
                out |= component
        return out
