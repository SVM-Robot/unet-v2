import csv
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window
# Suppress the specific warning
import warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
import torch
from torch.utils.data import Dataset
import random

def load_manifest_pairs(manifest_csv, data_root, split, classes=None):
    required_columns = {"split", "image_path", "mask_path", "class_name"}
    with open(manifest_csv, newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")
        rows = [
            row for row in reader
            if row["split"] == split and (classes is None or row["class_name"] in classes)
        ]

    root = Path(data_root)
    pairs = [
        (root / row["image_path"], root / row["mask_path"], row["class_name"])
        for row in rows
    ]
    missing = [path for image, mask, _ in pairs for path in (image, mask) if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[:5])
    return pairs


def read_tif(path: Path, bands: Optional[Sequence[int]] = None) -> np.ndarray:
    with rasterio.open(path) as src:
        if bands is None:
            array = src.read()
        else:
            array = src.read(indexes=list(bands))
    return array

def read_tif_window(path: Path, x: int, y: int, width: int, height: int, bands: Optional[Sequence[int]] = None) -> np.ndarray:
    

    window = Window(int(x), int(y), int(width), int(height))
    with rasterio.open(path) as src:
        if bands is None:
            return src.read(window=window)
        return src.read(indexes=list(bands), window=window)


_CLIP_MIN = np.array([-40.0, -30.0], dtype=np.float32)
_CLIP_MAX = np.array([-5.0, 0.0], dtype=np.float32)
_MEAN = np.array([-32.061756, -22.321358], dtype=np.float32)
_STD = np.array([7.5877995, 7.037356], dtype=np.float32)


def normalize_image(image: np.ndarray, bands: Optional[Sequence[int]] = None) -> np.ndarray:
    """Clip and standardize SAR channels. Band 1 is VH and band 2 is VV."""
    image = image.astype(np.float32)
    if bands is None:
        bands = tuple(range(1, image.shape[0] + 1))
    index = [band - 1 for band in bands]
    if image.ndim != 3 or image.shape[0] != len(index) or any(i not in (0, 1) for i in index):
        raise ValueError(
            f"Expected bands {list(bands)} in VH, VV order; got shape {image.shape}. "
            "Set --image-bands to the correct 1-based raster band indexes."
        )

    clip_min = _CLIP_MIN[index][:, None, None]
    clip_max = _CLIP_MAX[index][:, None, None]
    mean = _MEAN[index][:, None, None]
    std = _STD[index][:, None, None]
    return (np.clip(image, clip_min, clip_max) - mean) / (std + 1e-6)

def prepare_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)
    mask = np.squeeze(mask)
    mask = (mask > 0).astype(np.float32)
    return mask


class CropManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        data_root,
        sampling_config,
        image_bands: Optional[Sequence[int]] = (1, 2),
        output_dir=None,
    ):
        from src.crop_sampling import load_crop_rows, save_selected_manifest, select_training_crops

        self.data_root = Path(data_root)
        self.image_bands = image_bands
        self.rows = select_training_crops(load_crop_rows(manifest_csv), sampling_config)
        if output_dir is not None:
            save_selected_manifest(self.rows, output_dir, sampling_config, manifest_csv)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        x, y = int(row["x"]), int(row["y"])
        width, height = int(row["crop_width"]), int(row["crop_height"])
        image = normalize_image(read_tif_window(
            self.data_root / row["image_path"], x, y, width, height, bands=self.image_bands
        ), bands=self.image_bands)
        mask = prepare_mask(read_tif_window(
            self.data_root / row["mask_path"], x, y, width, height
        ))
        if image.shape[-2:] != (height, width) or mask.shape != (height, width):
            raise ValueError(f"Crop {row['image_id']} at {(x, y)} has image {image.shape} and mask {mask.shape}")
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0)



class OilSpillDataset(Dataset):
    def __init__(self, pairs, image_bands=None, crop_size=None, sample_mode=False, image_size=2048):
        self.pairs = list(pairs)
        self.image_bands = image_bands
        self.sample_mode = sample_mode

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path, class_name = self.pairs[index]
        image = normalize_image(read_tif(image_path, bands=self.image_bands), bands=self.image_bands)
        mask = prepare_mask(read_tif(mask_path))
        return torch.from_numpy(image).float(), torch.from_numpy(mask).float().unsqueeze(0), class_name
