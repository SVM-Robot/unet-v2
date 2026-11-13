from __future__ import annotations

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

from src.model import UNet4
from src.utils import dice_score, dice_score_oil, water_false_positive_rate, predict_full_image, dice_score_from_probs, iou_score_from_probs


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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet4(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            bi_upsample=bi_upsample,
        )

        # Moves with the module to the correct device automatically.
        self.register_buffer("pos_weight", torch.tensor([lwc]))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, dice_score_oil(logits, y), water_false_positive_rate(logits, y)

    def _log_split(self, split, loss, dice, water_fp, y):
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, sync_dist=split != "train")
        oil_tiles = int((y.sum(dim=(1, 2, 3)) > 0).sum())
        water_tiles = int((y.sum(dim=(1, 2, 3)) == 0).sum())
        if dice is not None:
            self.log(f"{split}_dice", dice, on_step=False, on_epoch=True, sync_dist=split != "train", batch_size=oil_tiles)
        if water_fp is not None:
            self.log(f"{split}_water_fp", water_fp, on_step=False, on_epoch=True, sync_dist=split != "train", batch_size=water_tiles)

    def _eval_full_image(self, image, mask):
        probs = predict_full_image(
            self, image, self.device,
            crop_size=self.hparams.infer_crop_size,
            overlap=self.hparams.infer_overlap,
        )
        mask = mask[0].cpu()
        dice = dice_score_from_probs(probs, mask[0])
        iou = iou_score_from_probs(probs, mask[0])
        return dice, iou


    def training_step(self, batch, batch_idx):
        loss, dice, water_fp = self._shared_step(batch)
        self._log_split("train", loss, dice, water_fp, batch[1])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        dice, iou = self._eval_full_image(x[0], y[0])
        self.log("val_dice", dice, on_epoch=True, sync_dist=True)
        self.log("val_iou", iou, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        dice, iou = self._eval_full_image(x[0], y[0])
        self.log("test_dice", dice, on_epoch=True, sync_dist=True)
        self.log("test_iou", iou, on_epoch=True, sync_dist=True)




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
                "monitor": "val_dice",
            },
        }
