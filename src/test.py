from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import OilSpillDataset, load_manifest_pairs
from src.lit_module import OilSpillLitModule
from src.utils import set_seed, dice_score_from_probs, iou_score_from_probs
from src.utils import predict_full_image


def load_model(checkpoint_path: Path, device: str, in_channels: int, args: argparse.Namespace) -> OilSpillLitModule:
    if checkpoint_path.suffix == ".ckpt":
        model = OilSpillLitModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
        )
        model.eval()
        return model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})

    model = OilSpillLitModule(
        in_channels=in_channels,
        base_channels=train_args.get("base_channels", args.base_channels),
        depth=train_args.get("depth", args.depth),
        batch_norm=train_args.get("batch_norm", args.batch_norm),
        dropout_rate=train_args.get("dropout_rate", args.dropout_rate),
        bi_upsample=train_args.get("bi_upsample", args.bi_upsample),
        lr=train_args.get("lr", 1e-3),
        lwc=train_args.get("lwc", 1),
        lr_scheduler=train_args.get("lr_scheduler", False),
    )
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained UNet and save visual results")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.ckpt (Lightning) or best.pt (legacy)")
    parser.add_argument("--output-dir", type=str, default="runs/unet_impl1/test_vis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--classes", type=str, nargs="*", default=["oil", "lookalike", "no_oil"])
    parser.add_argument("--image-bands", type=int, nargs="*", default=None)
    parser.add_argument("--base-channels", type=int, default=16, help="Fallback for legacy .pt checkpoints only")
    parser.add_argument("--depth", type=int, default=2, help="Fallback for legacy .pt checkpoints only")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--crop-size", type=int, default=-1, help="Run patch-based inference with this crop size.")
    parser.add_argument("--test-overlap", type=float, default=0.25, help="Overlap fraction used when running patch-based inference.")
    parser.add_argument("--batch-norm", action="store_true", default=False, help="Fallback for legacy .pt checkpoints only")
    parser.add_argument("--dropout-rate", type=float, default=0.0, help="Fallback for legacy .pt checkpoints only")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bi-upsample", action="store_true", default=False, help="Fallback for legacy .pt checkpoints only")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bands = args.image_bands
    test_pairs = load_manifest_pairs(args.manifest_csv, args.data_root, "test", args.classes)
    if not test_pairs:
        raise RuntimeError("The manifest has no samples in the test split for the selected classes")

    dataset = OilSpillDataset(
        test_pairs,
        image_bands=image_bands,
        sample_mode=True,
    )
    sample_image, _ = dataset[0]
    in_channels = sample_image.shape[0]
    device = args.device

    model = load_model(checkpoint_path, device, in_channels, args)
    hparams = model.hparams

    print("=== Test start ===\n")
    print(f"checkpoint: {checkpoint_path}")
    print(
        f"model config: depth={hparams.depth}, base_channels={hparams.base_channels}, "
        f"batch_norm={hparams.batch_norm}, dropout_rate={hparams.dropout_rate}, "
        f"bi_upsample={hparams.bi_upsample}"
    )
    print(f"data config: image_bands={image_bands}, normalization=Trujillo per-band")
    print("Using device:", device)

    total_dice = 0.0
    total_iou = 0.0
    if args.max_samples >= 0:
        tot_samples = min(len(dataset), args.max_samples)
    else:
        tot_samples = len(dataset)
    if tot_samples == 0:
        raise ValueError("No test samples selected; --max-samples must be positive")

    for i, (image, mask) in enumerate(dataset):
        if i >= tot_samples:
            break

        probs = predict_full_image(
            model,
            image,
            device=device,
            crop_size=args.crop_size,
            overlap=args.test_overlap,
        )
        mask_cpu = mask[0].cpu()

        sample_dice = float(dice_score_from_probs(probs, mask_cpu, threshold=args.threshold).item())
        total_dice += sample_dice

        sample_iou = float(iou_score_from_probs(probs, mask_cpu, threshold=args.threshold).item())
        total_iou += sample_iou

    mean_dice = total_dice / tot_samples
    mean_iou = total_iou / tot_samples
    summary = {"mean_test_dice": mean_dice, "mean_test_iou": mean_iou, "num_samples": tot_samples}

    with open(output_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n***\nTest Results:\n"
        f"mean test dice: {mean_dice:.4f}, mean test iou: {mean_iou:.4f}. "
        f"({tot_samples} test samples)\n***\n"
    )
    print("=== Test finished ===")


if __name__ == "__main__":
    main()
