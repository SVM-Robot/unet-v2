from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CropSamplingConfig:
    crop_size: int = 512
    stride: int = 512
    training_mode: str = "hard_negative"  # "oil_only" or "hard_negative"
    include_all_positive_crops: bool = True
    max_empty_crops_per_oil_image: int = 2
    max_crops_per_lk_image: int = 2
    max_crops_per_no_oil_image: int = 2
    seed: int = 42
    max_positive_crops_per_image: Optional[int] = None
    max_positive_crops_per_scene: Optional[int] = None


def candidate_origins(height: int, width: int, crop_size: int, stride: int):
    """Same grid as the master manifest, including a final boundary-aligned crop."""
    def axis(length: int):
        coords = list(range(0, length - crop_size + 1, stride))
        last = length - crop_size
        if not coords or coords[-1] != last:
            coords.append(last)
        return coords

    return [(x, y) for y in axis(height) for x in axis(width)]


def load_crop_rows(manifest_csv: str | Path) -> list[dict]:
    with open(manifest_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_id", "category", "scene_product_id", "image_path", "mask_path",
            "x", "y", "crop_width", "crop_height", "crop_size", "contains_oil",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Crop manifest is missing columns: {sorted(missing)}")
        return list(reader)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _take(groups: dict, k: Optional[int], rng: random.Random) -> list[dict]:
    chosen = []
    for key in sorted(groups):
        pool = groups[key]
        n = len(pool) if k is None else min(int(k), len(pool))
        if n <= 0:
            continue
        if n == len(pool):
            chosen.extend(pool)
        else:
            chosen.extend(rng.sample(list(pool), n))
    return chosen


def select_training_crops(rows: list[dict], config: CropSamplingConfig) -> list[dict]:
    if config.training_mode not in {"oil_only", "hard_negative"}:
        raise ValueError(f"Unknown training_mode {config.training_mode!r}")

    sizes = {int(row["crop_size"]) for row in rows}
    if sizes != {config.crop_size}:
        raise ValueError(f"Manifest crop_size {sizes} does not match config {config.crop_size}")

    positives_by_image = defaultdict(list)
    oil_empty = defaultdict(list)
    lookalike = defaultdict(list)
    no_oil = defaultdict(list)

    for row in rows:
        if "split" in row and row["split"] not in {"", "train"}:
            continue
        item = dict(row)
        category = row["category"]
        has_oil = _as_bool(row["contains_oil"])
        if category == "oil" and has_oil:
            item["sample_role"] = "positive"
            positives_by_image[row["image_id"]].append(item)
        elif category == "oil":
            item["sample_role"] = "oil_empty"
            oil_empty[row["image_id"]].append(item)
        elif category == "lookalike":
            item["sample_role"] = "lookalike"
            lookalike[row["image_id"]].append(item)
        elif category == "no_oil":
            item["sample_role"] = "no_oil"
            no_oil[row["image_id"]].append(item)
        else:
            raise ValueError(f"Unknown category {category!r}")

    pos_rng = random.Random(f"{config.seed}:positives")
    if config.include_all_positive_crops and config.max_positive_crops_per_image is None:
        positives = []
        for image_id in sorted(positives_by_image):
            positives.extend(positives_by_image[image_id])
    else:
        positives = _take(positives_by_image, config.max_positive_crops_per_image, pos_rng)

    if config.max_positive_crops_per_scene is not None:
        by_scene = defaultdict(list)
        for row in positives:
            by_scene[row["scene_product_id"]].append(row)
        positives = _take(by_scene, config.max_positive_crops_per_scene, pos_rng)

    if config.training_mode == "oil_only":
        return positives

    # One generator, image ids sorted, oil empties then lookalikes then no-oil.
    neg_rng = random.Random(config.seed)
    selected = list(positives)
    selected.extend(_take(oil_empty, config.max_empty_crops_per_oil_image, neg_rng))
    selected.extend(_take(lookalike, config.max_crops_per_lk_image, neg_rng))
    selected.extend(_take(no_oil, config.max_crops_per_no_oil_image, neg_rng))
    return selected


def save_selected_manifest(rows: list[dict], output_dir: str | Path, config: CropSamplingConfig, source_csv: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "selected_crop_manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else ["sample_role"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = asdict(config)
    payload["source_crop_manifest"] = str(source_csv)
    payload["selected_crops"] = len(rows)
    (output_dir / "sampling_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path


def summarize_crops(rows: list[dict]) -> str:
    roles = Counter(row.get("sample_role", row["category"]) for row in rows)
    by_image = Counter(row["image_id"] for row in rows if row.get("sample_role") == "positive")
    by_scene = Counter(row["scene_product_id"] for row in rows if row.get("sample_role") == "positive")
    pos_counts = sorted(by_image.values())
    scene_counts = sorted(by_scene.values())
    lines = [
        f"selected={len(rows)} " + " ".join(f"{name}={roles[name]}" for name in sorted(roles)),
        f"positive images={len(by_image)} scenes={len(by_scene)}",
    ]
    if pos_counts:
        lines.append(
            f"positives/image min={pos_counts[0]} median={pos_counts[len(pos_counts)//2]} max={pos_counts[-1]}"
        )
        lines.append(
            f"positives/scene min={scene_counts[0]} median={scene_counts[len(scene_counts)//2]} max={scene_counts[-1]}"
        )
    return "\n".join(lines)