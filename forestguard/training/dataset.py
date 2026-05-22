"""PyTorch Dataset for paired before/after fused patches with deforestation labels."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


def default_augmentation() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        ],
        additional_targets={"before": "image", "label": "mask"},
    )


class ForestChangeDataset(Dataset):
    """
    Expects a directory containing JSON manifest with entries:
        {
          "before": "path/to/before_patch.npy",
          "after":  "path/to/after_patch.npy",
          "label":  "path/to/label_patch.npy"   // binary uint8 (H, W)
        }

    Arrays are (C, H, W) float32 in [0, 1].
    """

    def __init__(
        self,
        manifest_path: str | Path,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.augment = augment
        self.transform = transform or (default_augmentation() if augment else None)

        with open(self.manifest_path) as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.samples[idx]
        before = np.load(self.root / entry["before"]).astype(np.float32)  # (C, H, W)
        after = np.load(self.root / entry["after"]).astype(np.float32)
        label = np.load(self.root / entry["label"]).astype(np.float32)    # (H, W)

        if self.transform is not None:
            before, after, label = self._apply_transform(before, after, label)
        else:
            before = torch.from_numpy(before)
            after = torch.from_numpy(after)
            label = torch.from_numpy(label)

        return before, after, label.unsqueeze(0)  # label: (1, H, W)

    # ------------------------------------------------------------------
    def _apply_transform(
        self,
        before: np.ndarray,
        after: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # albumentations expects (H, W, C) for images
        before_hwc = before.transpose(1, 2, 0)
        after_hwc = after.transpose(1, 2, 0)
        label_hw = label.astype(np.uint8)

        result = self.transform(image=after_hwc, before=before_hwc, label=label_hw)

        before_t = torch.from_numpy(result["before"].transpose(2, 0, 1))
        after_t = torch.from_numpy(result["image"].transpose(2, 0, 1))
        label_t = torch.from_numpy(result["label"].astype(np.float32))
        return before_t, after_t, label_t

    @staticmethod
    def build_manifest(
        patches_dir: str | Path,
        output_path: str | Path,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        """Scan a patches directory and write a split manifest JSON."""
        patches_dir = Path(patches_dir)
        befores = sorted((patches_dir / "before").glob("*.npy"))
        samples = []
        for b in befores:
            stem = b.stem
            a = patches_dir / "after" / f"{stem}.npy"
            lbl = patches_dir / "labels" / f"{stem}.npy"
            if a.exists() and lbl.exists():
                samples.append(
                    {
                        "before": str(b.relative_to(patches_dir.parent)),
                        "after": str(a.relative_to(patches_dir.parent)),
                        "label": str(lbl.relative_to(patches_dir.parent)),
                    }
                )

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(samples))
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        splits = {
            "train": idx[:n_train],
            "val": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }
        chosen = [samples[i] for i in splits[split]]
        with open(output_path, "w") as f:
            json.dump(chosen, f, indent=2)
