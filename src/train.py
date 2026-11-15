from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.lit_datamodule import OilSpillDataModule
from src.lit_module import OilSpillLitModule
from src.crop_sampling import CropSamplingConfig, summarize_crops, load_crop_rows, select_training_crops


class EpochPrinter(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        print(
            f"epoch={trainer.current_epoch + 1:03d} "
            f"train_loss={m.get('train_loss', float('nan')):.4f} "
            f"train_dice={m.get('train_dice', float('nan')):.4f} "
            f"val_oil_dice={m.get('val_oil_dice', float('nan')):.4f} "
            f"val_oil_iou={m.get('val_oil_iou', float('nan')):.4f}"
        )

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple UNet for SAR oil spill segmentation")
    parser.add_argument("--manifest-csv", type=str, required=True)
    parser.add_argument("--crop-size", type=int, default=512, help="Training and inference window size.")
    parser.add_argument("--crop-manifest-csv", type=str, required=True)
    parser.add_argument("--training-mode", choices=["oil_only", "hard_negative"], default="hard_negative")
    parser.add_argument("--max-empty-crops-per-oil-image", type=int, default=2)
    parser.add_argument("--max-crops-per-lk-image", type=int, default=2)
    parser.add_argument("--max-crops-per-no-oil-image", type=int, default=2)
    parser.add_argument("--max-positive-crops-per-image", type=int, default=None)
    parser.add_argument("--max-positive-crops-per-scene", type=int, default=None)
    parser.add_argument("--infer-overlap", type=float, default=0.25)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--classes", type=str, nargs="*", default=["oil", "lookalike", "no_oil"])
    parser.add_argument("--output-dir", type=str, default="runs/unet_impl1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-bands", type=int, nargs="*", default=None, help="List of image bands to use (1-based indexing). If not specified, all bands will be used.")
    parser.add_argument("--single-crop", action="store_true", default=False, help="If set, only a single crop will be taken from each image. Otherwise, multiple crops will be taken.")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4, help="UNet depth")
    parser.add_argument("--device", type=str, default=None, help="Device to use for training (e.g., 'cuda', 'cpu', 'mps'). If not specified, will use 'cuda' if available, otherwise 'cpu'.")
    parser.add_argument("--lwc", type=int, default=1)
    parser.add_argument("--batch-norm", action="store_true", default=False)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--lr-scheduler", action="store_true", default=False)
    parser.add_argument("--bi-upsample", action="store_true", default=False)
    return parser.parse_args()


def _resolve_accelerator(device: str | None) -> tuple[str, int]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "gpu", 1
    if device == "mps":
        return "mps", 1
    return "cpu", 1


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampling = CropSamplingConfig(
        crop_size=args.crop_size,
        stride=args.crop_size,
        training_mode=args.training_mode,
        max_empty_crops_per_oil_image=args.max_empty_crops_per_oil_image,
        max_crops_per_lk_image=args.max_crops_per_lk_image,
        max_crops_per_no_oil_image=args.max_crops_per_no_oil_image,
        seed=args.seed,
        max_positive_crops_per_image=args.max_positive_crops_per_image,
        max_positive_crops_per_scene=args.max_positive_crops_per_scene,
    )
    print(summarize_crops(select_training_crops(load_crop_rows(args.crop_manifest_csv), sampling)))
    datamodule = OilSpillDataModule(
        manifest_csv=args.manifest_csv,
        crop_manifest_csv=args.crop_manifest_csv,
        data_root=args.data_root,
        sampling_config=sampling,
        output_dir=output_dir,
        classes=args.classes,
        batch_size=args.batch_size,
        image_bands=args.image_bands,
    )
    datamodule.setup()

    model = OilSpillLitModule(
        in_channels=datamodule.in_channels,
        base_channels=args.base_channels,
        depth=args.depth,
        batch_norm=args.batch_norm,
        dropout_rate=args.dropout_rate,
        bi_upsample=args.bi_upsample,
        lr=args.lr,
        lwc=args.lwc,
        lr_scheduler=args.lr_scheduler,
        infer_crop_size=sampling.crop_size,
        infer_overlap=args.infer_overlap,
    )

    accelerator, devices = _resolve_accelerator(args.device)
    print("Using accelerator:", accelerator)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best",
            monitor="val_oil_dice",
            mode="max",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
        ),
        EpochPrinter(),
    ]
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_oil_dice",
                mode="max",
                patience=10,
            )
        )

    loggers = [
        CSVLogger(save_dir=output_dir, name="logs"),
        TensorBoardLogger(save_dir=output_dir, name="tb_logs"),
    ]

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        default_root_dir=output_dir,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            print("TensorBoard logs:", logger.log_dir)

    trainer.fit(model, datamodule)

    checkpoint_callback = trainer.checkpoint_callback

    if checkpoint_callback.best_model_path:
        print(
            f"Best checkpoint: {checkpoint_callback.best_model_path} "
            f"(val_oil_dice={checkpoint_callback.best_model_score:.4f})."
        )





if __name__ == "__main__":
    main()
