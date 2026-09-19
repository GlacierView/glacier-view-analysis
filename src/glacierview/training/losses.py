"""Loss functions.

Cross-entropy is the published choice: it penalises how wrong a prediction is,
not just whether it is wrong, which suits blurry ice/rock boundaries. Dice
loss is kept because the paper compared against it (0.90 vs 0.92 Dice).

The Dice implementations are adapted from a commonly circulated reference
implementation; the original script credited it only as "source".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert a class-index tensor [N, 1, *] to one-hot [N, num_classes, *]."""
    shape = np.array(labels.shape)
    shape[1] = num_classes
    result = torch.zeros(tuple(shape))
    return result.scatter_(1, labels.cpu(), 1)


class BinaryDiceLoss(nn.Module):
    """Dice loss for a single class.

    Args:
        smooth: added to numerator and denominator to avoid NaN.
        p: denominator exponent, ``sum(x^p) + sum(y^p)``.
        reduction: ``mean``, ``sum`` or ``none``.
    """

    def __init__(self, smooth: float = 1, p: int = 2, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predict.shape[0] != target.shape[0]:
            raise ValueError("predict and target batch sizes differ")
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"unknown reduction {self.reduction!r}")


class DiceLoss(nn.Module):
    """Multi-class Dice loss over one-hot targets."""

    def __init__(self, weight=None, ignore_index: int | None = None, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if predict.shape != target.shape:
            raise ValueError("predict and target shapes differ")
        dice = BinaryDiceLoss(**self.kwargs)
        predict = F.softmax(predict, dim=1)

        total = 0.0
        for i in range(target.shape[1]):
            if i == self.ignore_index:
                continue
            loss = dice(predict[:, i], target[:, i])
            if self.weight is not None:
                if self.weight.shape[0] != target.shape[1]:
                    raise ValueError(
                        f"expected weight shape [{target.shape[1]}], "
                        f"got [{self.weight.shape[0]}]"
                    )
                loss = loss * self.weight[i]
            total = total + loss
        return total / target.shape[1]


def build_criterion(name: str):
    """Map a loss name to a callable. ``combo`` sums cross-entropy and Dice."""
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return BinaryDiceLoss()
    if name == "mse":
        return nn.MSELoss()
    if name == "combo":
        ce, dice = nn.CrossEntropyLoss(), BinaryDiceLoss()
        return lambda out, seg: ce(out, seg) + dice(out, seg)
    raise ValueError(f"unknown loss {name!r}; expected ce, dice, mse or combo")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
# Computed here rather than via torchmetrics: `torchmetrics.Dice` was removed
# in 1.9 (it is now `segmentation.DiceScore`, with a different signature), and
# these are two-line formulas. Implementing them directly keeps the metric
# stable across torchmetrics releases and makes it checkable by hand.

def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Sorensen-Dice coefficient: 2|A and B| / (|A| + |B|), in [0, 1].

    Both inputs are treated as binary masks. Plain accuracy is useless for
    this task — a glacier covering 5% of the frame scores 95% by predicting no
    ice at all — which is why Dice is the reported metric.
    """
    pred = (pred > 0.5).flatten().float()
    target = (target > 0.5).flatten().float()
    overlap = (pred * target).sum()
    return float((2 * overlap + eps) / (pred.sum() + target.sum() + eps))


def jaccard_index(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Intersection over union: |A and B| / |A or B|, in [0, 1]."""
    pred = (pred > 0.5).flatten().float()
    target = (target > 0.5).flatten().float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))
