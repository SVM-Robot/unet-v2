#!/usr/bin/env python3
"""Pre-extract 512x512 training crops to scratch for fast DataLoader I/O."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def crop_key(image_id: str, x: int, y: int) -> str:
    safe = image_id.replace("/", "__")
    return f"{safe}__x{x}_y{y}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crop-manifest-csv", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--output-dir",
        default="/cluster/work/projects/nn8106k/svm/crop_cache_512",
        help="Write arrays + index here (prefer scratch/work, not project data/)",
    )
    p.add_argument("--image-bands", type=int, nargs="*", default=[1, 2])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    img_dir = out / "images"
    mask_dir = out / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with open(args.crop_manifest_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_image = defaultdict(list)
    for i, row in enumerate(rows):
        by_image[row["image_path"]].append((i, row))

    index_path = out / "crop_index.csv"
    fieldnames = list(rows[0].keys()) + [
        "cache_key",
        "cache_image_path",
        "cache_mask_path",
        "cache_bands",
    ]

    written = 0
    skipped = 0
    n_images = len(by_image)

    with index_path.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for img_i, (image_rel, items) in enumerate(by_image.items(), start=1):
            if img_i == 1 or img_i % 50 == 0 or img_i == n_images:
                print(f"images {img_i}/{n_images} wrote={written} skipped={skipped}", flush=True)

            image_path = data_root / image_rel
            mask_rel = items[0][1]["mask_path"]
            mask_path = data_root / mask_rel

            with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
                for _, row in items:
                    x, y = int(row["x"]), int(row["y"])
                    w, h = int(row["crop_width"]), int(row["crop_height"])
                    key = crop_key(row["image_id"], x, y)
                    img_out = img_dir / f"{key}.npy"
                    mask_out = mask_dir / f"{key}.npy"

                    if img_out.exists() and mask_out.exists() and not args.overwrite:
                        skipped += 1
                    else:
                        window = Window(x, y, w, h)
                        image = img_src.read(indexes=list(args.image_bands), window=window)
                        mask = np.squeeze(mask_src.read(window=window))
                        # raw float32 image (NOT normalized); binary uint8 mask
                        np.save(img_out, image.astype(np.float32))
                        np.save(mask_out, (mask > 0).astype(np.uint8))
                        written += 1

                    out_row = dict(row)
                    out_row.update(
                        {
                            "cache_key": key,
                            "cache_image_path": str(img_out.relative_to(out)),
                            "cache_mask_path": str(mask_out.relative_to(out)),
                            "cache_bands": " ".join(map(str, args.image_bands)),
                        }
                    )
                    writer.writerow(out_row)

    meta = out / "cache_meta.txt"
    meta.write_text(
        f"source_manifest={args.crop_manifest_csv}\n"
        f"data_root={args.data_root}\n"
        f"bands={args.image_bands}\n"
        f"n_rows={len(rows)}\n"
        f"written={written}\n"
        f"skipped_existing={skipped}\n"
        f"note=images are RAW float32 (apply normalize_image at train time)\n",
        encoding="utf-8",
    )
    print(f"Done. wrote={written} skipped={skipped} index={index_path}")


if __name__ == "__main__":
    main()