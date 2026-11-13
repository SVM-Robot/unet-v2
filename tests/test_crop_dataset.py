import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.crop_sampling import CropSamplingConfig, candidate_origins, select_training_crops
from src.data import prepare_mask


def _rows():
    rows = []
    for category, image_id in (("oil", "oil_a"), ("lookalike", "lk_a"), ("no_oil", "no_a")):
        for x in (0, 512):
            for y in (0, 512):
                oil = category == "oil" and (x, y) == (0, 0)
                rows.append({
                    "image_id": image_id,
                    "category": category,
                    "scene_product_id": "scene-" + category,
                    "image_path": image_id + ".tif",
                    "mask_path": image_id + "_mask.tif",
                    "x": x, "y": y,
                    "crop_width": 512, "crop_height": 512, "crop_size": 512,
                    "contains_oil": oil,
                    "split": "train",
                })
    return rows


def test_grid_covers_2048():
    origins = candidate_origins(2048, 2048, 512, 512)
    assert len(origins) == 16
    covered = np.zeros((2048, 2048), dtype=np.int32)
    for x, y in origins:
        covered[y:y + 512, x:x + 512] += 1
    assert covered.min() == 1


def test_policies_share_positives_and_seed():
    cfg = dict(crop_size=512, stride=512, seed=42, max_empty_crops_per_oil_image=2,
               max_crops_per_lk_image=2, max_crops_per_no_oil_image=2)
    oil = select_training_crops(_rows(), CropSamplingConfig(training_mode="oil_only", **cfg))
    hard = select_training_crops(_rows(), CropSamplingConfig(training_mode="hard_negative", **cfg))
    again = select_training_crops(_rows(), CropSamplingConfig(training_mode="hard_negative", **cfg))
    assert [row["sample_role"] for row in oil] == ["positive"]
    assert [row for row in hard if row["sample_role"] == "positive"] == oil
    assert Counter(row["sample_role"] for row in hard)["lookalike"] == 2
    assert hard == again


def test_contains_oil_matches_prepared_mask():
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    mask[0, 0, 1] = 3
    prepared = prepare_mask(mask)
    assert prepared.shape == (4, 4)
    assert bool(prepared.sum() > 0)


def test_dataloader_batches_single_crops():
    class _Fake(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, index):
            image = torch.zeros(2, 512, 512)
            mask = torch.zeros(1, 512, 512)
            return image, mask

    images, masks = next(iter(DataLoader(_Fake(), batch_size=2)))
    assert images.shape == (2, 2, 512, 512)
    assert masks.shape == (2, 1, 512, 512)