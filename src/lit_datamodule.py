from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.crop_sampling import CropSamplingConfig
from src.data import CropManifestDataset, OilSpillDataset, load_manifest_pairs


class OilSpillDataModule(LightningDataModule):
    def __init__(
        self,
        manifest_csv: str | Path,
        crop_manifest_csv: str | Path,
        data_root: str | Path,
        sampling_config: CropSamplingConfig,
        output_dir: str | Path,
        classes: Optional[Sequence[str]] = None,
        batch_size: int = 2,
        image_bands: Optional[Sequence[int]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        crop_cache_dir: str | Path | None = None,
    ):
        super().__init__()
        self.manifest_csv = manifest_csv
        self.crop_manifest_csv = crop_manifest_csv
        self.data_root = data_root
        self.sampling_config = sampling_config
        self.output_dir = output_dir
        self.classes = classes
        self.batch_size = batch_size
        self.image_bands = image_bands
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.crop_cache_dir = crop_cache_dir
        self.val_pairs = None
        self.test_pairs = None
        self.in_channels: Optional[int] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None) and self.val_pairs is None:
            self.val_pairs = load_manifest_pairs(self.manifest_csv, self.data_root, "val", self.classes)
            if not self.val_pairs:
                raise RuntimeError("No validation images for the selected classes")
        if stage in ("test", None) and self.test_pairs is None:
            self.test_pairs = load_manifest_pairs(self.manifest_csv, self.data_root, "test", self.classes)
            if not self.test_pairs:
                raise RuntimeError("No test images for the selected classes")
        if self.in_channels is None:
            probe = OilSpillDataset(self.val_pairs or self.test_pairs, image_bands=self.image_bands, sample_mode=True)
            image, _, _ = probe[0]
            self.in_channels = image.shape[0]

    def train_dataloader(self) -> DataLoader:
        dataset = CropManifestDataset(
            self.crop_manifest_csv,
            self.data_root,
            self.sampling_config,
            image_bands=self.image_bands,
            output_dir=self.output_dir,
            crop_cache_dir=self.crop_cache_dir,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def _full_image_loader(self, pairs) -> DataLoader:
        dataset = OilSpillDataset(pairs, image_bands=self.image_bands, sample_mode=True)
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return self._full_image_loader(self.val_pairs)

    def test_dataloader(self) -> DataLoader:
        return self._full_image_loader(self.test_pairs)
