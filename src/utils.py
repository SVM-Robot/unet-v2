import numpy as np
import torch
import random
import math
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def dice_score(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()

def iou_score(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    sum_sets = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    union = sum_sets - intersection

    return ((intersection + eps) / (union + eps)).mean()


def dice_score_oil(logits, targets, threshold: float = 0.5, eps: float = 1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()
    oil = targets.sum(dim=(1, 2, 3)) > 0
    if int(oil.sum()) == 0:
        return None
    preds = preds[oil]
    targets = targets[oil]
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (denom + eps)).mean()


def water_false_positive_rate(logits, targets, threshold: float = 0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()
    water = targets.sum(dim=(1, 2, 3)) == 0
    if int(water.sum()) == 0:
        return None
    return preds[water].mean()


def dice_score_from_probs(probs, targets, threshold: float = 0.5, eps: float = 1e-6):
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2 * intersection + eps) / (union + eps)


def iou_score_from_probs(probs, targets, threshold: float = 0.5, eps: float = 1e-6):
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)



def predict_full_image(model, image, device, crop_size=-1, overlap=0.0, wind=None):
    image = image.float()
    if wind is not None:
        wind = wind.float()

    def logits_of(patch):
        wind_batch = None if wind is None else wind.unsqueeze(0).to(device)
        with torch.no_grad():
            return model(patch.unsqueeze(0).to(device), wind_batch)

    if crop_size <= 0:
        logits = logits_of(image)
        return torch.sigmoid(logits)[0, 0].cpu()

    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0, 1)")

    h, w = image.shape[1], image.shape[2]
    stride = max(1, int(round(crop_size * (1.0 - overlap))))

    pad_h = max(0, (math.ceil(h / stride) * stride) - h)
    pad_w = max(0, (math.ceil(w / stride) * stride) - w)

    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h)).squeeze(0)
    padded_h, padded_w = padded.shape[1], padded.shape[2]

    full_probs = torch.zeros((h, w), dtype=torch.float32)
    counts = torch.zeros((h, w), dtype=torch.float32)

    for top in range(0, padded_h, stride):
        for left in range(0, padded_w, stride):
            patch = padded[:, top : top + crop_size, left : left + crop_size]

            if patch.shape[1] < crop_size or patch.shape[2] < crop_size:
                patch = F.pad(
                    patch,
                    (0, crop_size - patch.shape[2], 0, crop_size - patch.shape[1]),
                )

            logits = logits_of(patch)
            probs = torch.sigmoid(logits)[0, 0].cpu()

            valid_h = min(crop_size, h - top)
            valid_w = min(crop_size, w - left)

            if valid_h <= 0 or valid_w <= 0:
                continue

            full_probs[top : top + valid_h, left : left + valid_w] += probs[:valid_h, :valid_w]
            counts[top : top + valid_h, left : left + valid_w] += 1.0

    full_probs = full_probs / counts.clamp_min(1.0)
    return full_probs