from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from src.model import UNet4
from src.utils import dice_score_oil, water_false_positive_rate, predict_full_image, dice_score_from_probs, iou_score_from_probs
from src.wind import WIND_MODES


class OilSpillLitModule(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
        bi_upsample: bool = False,
        lr: float = 1e-3,
        lwc: float = 1.0,
        lr_scheduler: bool = False,
        infer_crop_size: int = 512,
        infer_overlap: float = 0.25,
        loss_type: str = "bce",
        wind_mode: str = "none",
        wind_dim: int = 0,
        wind_norm: str = "normal",
        wind_mean: list[float] | None = None,
        wind_std: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if wind_mode not in WIND_MODES:
            raise ValueError(f"Unsupported wind_mode {wind_mode!r}")
        # Older checkpoints stored these buffers. Normalization now happens in WindTable.
        if wind_mean is not None and wind_std is not None:
            self.register_buffer("wind_mean", torch.tensor(wind_mean, dtype=torch.float32))
            self.register_buffer("wind_std", torch.tensor(wind_std, dtype=torch.float32))

        self.model = UNet4(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            bi_upsample=bi_upsample,
            wind_mode=wind_mode if wind_mode.startswith("film") else "none",
            wind_dim=wind_dim,
        )

        # Moves with the module to the correct device automatically.
        self.register_buffer("pos_weight", torch.tensor([lwc]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self._val_oil_dice = []
        self._val_lk_area = []
        self._val_no_oil_area = []

    def forward(self, x: torch.Tensor, wind: torch.Tensor | None = None) -> torch.Tensor:
        if self.hparams.wind_mode == "point_channels":
            if wind is None:
                raise ValueError("point_channels model requires a wind vector")
            wind = wind[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, wind], dim=1)
            return self.model(x)
        if self.hparams.wind_mode.startswith("film"):
            if wind is None:
                raise ValueError(f"{self.hparams.wind_mode} requires a wind tensor")
            return self.model(x, wind)
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce = self.criterion(logits, y)
        if self.hparams.loss_type == "bce":
            return bce
        if self.hparams.loss_type == "bce_dice":
            probs = torch.sigmoid(logits)
            intersection = (probs * y).sum()
            denom = probs.sum() + y.sum()
            soft_dice = (2.0 * intersection + 1.0) / (denom + 1.0)
            return 0.5 * bce + 0.5 * (1.0 - soft_dice)
        raise ValueError(f"Unknown loss_type {self.hparams.loss_type!r}")

    def _shared_step(self, batch):
        if isinstance(batch, dict):
            x, wind, y = batch["image"], batch["wind"], batch["mask"]
        else:
            x, y = batch
            wind = None
        logits = self(x, wind)
        loss = self._compute_loss(logits, y)
        return loss, dice_score_oil(logits, y), water_false_positive_rate(logits, y), y

    def _log_split(self, split, loss, dice, water_fp, y):
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, sync_dist=split != "train")
        oil_tiles = int((y.sum(dim=(1, 2, 3)) > 0).sum())
        water_tiles = int((y.sum(dim=(1, 2, 3)) == 0).sum())
        if dice is not None:
            self.log(f"{split}_dice", dice, on_step=False, on_epoch=True, sync_dist=split != "train", batch_size=oil_tiles)
        if water_fp is not None:
            self.log(f"{split}_water_fp", water_fp, on_step=False, on_epoch=True, sync_dist=split != "train", batch_size=water_tiles)

    def _eval_full_image(self, image, mask, wind=None):
        probs = predict_full_image(
            self, image, self.device,
            crop_size=self.hparams.infer_crop_size,
            overlap=self.hparams.infer_overlap,
            wind=wind,
        )
        mask_cpu = mask[0].cpu()
        return probs, mask_cpu

    def _log_full_image_metrics(self, split, image, mask, class_name, wind=None):
        probs, target = self._eval_full_image(image, mask, wind)
        prediction = probs > 0.5
        predicted_area = prediction.float().mean()
        if class_name == "oil":
            if target.sum() <= 0:
                raise ValueError("Manifest labels an image as oil but its mask is empty")
            dice = dice_score_from_probs(probs, target)
            iou = iou_score_from_probs(probs, target)
            self.log(f"{split}_oil_dice", dice, on_epoch=True, sync_dist=True, batch_size=1)
            self.log(f"{split}_oil_iou", iou, on_epoch=True, sync_dist=True, batch_size=1)
            if split == "val":
                self._val_oil_dice.append(dice.detach())
        elif class_name in {"lookalike", "no_oil"}:
            if target.sum() > 0:
                raise ValueError(f"Manifest labels {class_name!r} image but its mask contains oil")
            false_positive = prediction.any().float()
            self.log(f"{split}_{class_name}_fp_rate", false_positive, on_epoch=True, sync_dist=True, batch_size=1)
            self.log(f"{split}_{class_name}_predicted_oil_area", predicted_area, on_epoch=True, sync_dist=True, batch_size=1)
            if split == "val":
                if class_name == "lookalike":
                    self._val_lk_area.append(predicted_area.detach())
                else:
                    self._val_no_oil_area.append(predicted_area.detach())
        else:
            raise ValueError(f"Unsupported manifest class {class_name!r}")


    def training_step(self, batch, batch_idx):
        loss, dice, water_fp, y = self._shared_step(batch)
        self._log_split("train", loss, dice, water_fp, y)
        return loss

    def _full_image_example(self, batch):
        if isinstance(batch, dict):
            wind = batch["wind"][0] if "wind" in batch else None
            return batch["image"][0], batch["mask"][0], batch["class_name"][0], wind
        x, y, class_names = batch
        return x[0], y[0], class_names[0], None

    def validation_step(self, batch, batch_idx):
        image, mask, class_name, wind = self._full_image_example(batch)
        self._log_full_image_metrics("val", image, mask, class_name, wind)

    def on_validation_epoch_end(self):
        oil_dice = (
            torch.stack(self._val_oil_dice).mean()
            if self._val_oil_dice
            else torch.tensor(0.0, device=self.device)
        )
        lk_area = (
            torch.stack(self._val_lk_area).mean()
            if self._val_lk_area
            else torch.tensor(0.0, device=self.device)
        )
        no_oil_area = (
            torch.stack(self._val_no_oil_area).mean()
            if self._val_no_oil_area
            else torch.tensor(0.0, device=self.device)
        )

        val_score = oil_dice - 0.5 * (lk_area + no_oil_area)
        self.log("val_score", val_score, prog_bar=False, sync_dist=True)

        self._val_oil_dice.clear()
        self._val_lk_area.clear()
        self._val_no_oil_area.clear()

    def test_step(self, batch, batch_idx):
        image, mask, class_name, wind = self._full_image_example(batch)
        self._log_full_image_metrics("test", image, mask, class_name, wind)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        if not self.hparams.lr_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_score",
            },
        }
