"""The training loop, as functions rather than module-level script body.

Two defects in the original loop are fixed here, and both change training
behaviour — the published checkpoint was produced *with* them:

1. ``optimizer.zero_grad()`` was never called, so gradients accumulated across
   every batch in an epoch instead of being applied per step. With Adam that
   is not a strategy, it is a bug.
2. Only the final batch's loss was recorded per epoch, so the loss curve was
   a single noisy sample rather than an epoch mean.

Reproducing the paper's numbers exactly therefore requires the old script;
see git history. New runs should use this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glacierview import config
from glacierview.log import get_logger
from glacierview.models import UNet
from glacierview.training.losses import (
    build_criterion,
    dice_score,
    jaccard_index,
    make_one_hot,
)

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    """Hyperparameters. Defaults are the published recipe, not the old
    script's defaults — notably batch size 32 rather than 2."""

    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    batch_size: int = 32
    loss: str = "ce"
    freeze_encoder: bool = False
    unfreeze_after_epoch: int | None = None
    shuffle: bool = True
    augment: bool = True
    normalize: bool = True
    seed: int | None = 0
    num_workers: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class TrainResult:
    epoch_losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)
    best_epoch: int | None = None
    dice: float | None = None
    jaccard: float | None = None


def _prepare_targets(segs: torch.Tensor, loss_name: str, device) -> torch.Tensor:
    """Shape the mask for the chosen loss."""
    if loss_name == "dice":
        return make_one_hot(segs.long(), 2).to(device)
    return segs.squeeze(1).to(device).long()


def run_epoch(model, loader, criterion, optimizer, device, loss_name: str) -> float:
    """One training pass. Returns the mean batch loss."""
    model.train()
    total, batches = 0.0, 0
    for imgs, segs in tqdm(loader, leave=False):
        optimizer.zero_grad()          # the original never did this
        outputs = model(imgs.to(device))
        loss = criterion(outputs, _prepare_targets(segs, loss_name, device))
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, loss_name: str) -> float:
    """Mean loss over a held-out set, used to pick the best epoch."""
    model.eval()
    total, batches = 0.0, 0
    for imgs, segs in loader:
        outputs = model(imgs.to(device))
        total += criterion(outputs, _prepare_targets(segs, loss_name, device)).item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def score(model, loader, device) -> tuple[float, float]:
    """Mean Dice and Jaccard over a loader.

    Plain accuracy is useless here: a glacier covering 5% of the frame scores
    95% by predicting no ice at all.
    """
    model.eval()
    dices, jaccards = [], []
    for imgs, segs in tqdm(loader, leave=False):
        logits = model(imgs.to(device))
        preds = (torch.softmax(logits, dim=1)[:, 1] > config.PROB_THRESHOLD).float()
        target = segs.to(device).squeeze(1)
        dices.append(dice_score(preds, target))
        jaccards.append(jaccard_index(preds, target))
    n = max(len(dices), 1)
    return sum(dices) / n, sum(jaccards) / n


def fit(
    train_ds,
    eval_ds,
    test_ds,
    cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
) -> tuple[UNet, TrainResult]:
    """Train, tracking the eval set, and keep the best epoch's weights."""
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size,
                            shuffle=cfg.shuffle, num_workers=cfg.num_workers),
        "eval": DataLoader(eval_ds, batch_size=cfg.batch_size,
                           num_workers=cfg.num_workers),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size,
                           num_workers=cfg.num_workers),
    }

    model = UNet(n_class=2, freeze_encoder=cfg.freeze_encoder).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    criterion = build_criterion(cfg.loss)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = TrainResult()
    best_loss, best_state = float("inf"), None

    for epoch in range(cfg.epochs):
        if cfg.unfreeze_after_epoch is not None and epoch > cfg.unfreeze_after_epoch:
            for param in model.resnet.parameters():
                param.requires_grad = True

        train_loss = run_epoch(
            model, loaders["train"], criterion, optimizer, device, cfg.loss
        )
        eval_loss = evaluate_loss(model, loaders["eval"], criterion, device, cfg.loss)
        result.epoch_losses.append(train_loss)
        result.eval_losses.append(eval_loss)
        logger.info(
            "epoch %d/%d  train %.4f  eval %.4f", epoch + 1, cfg.epochs,
            train_loss, eval_loss,
        )

        # The published procedure keeps the epoch that minimises eval loss,
        # not whatever the last epoch happened to produce.
        if eval_loss < best_loss:
            best_loss, result.best_epoch = eval_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("restored epoch %d (eval loss %.4f)", result.best_epoch + 1, best_loss)

    result.dice, result.jaccard = score(model, loaders["test"], device)
    logger.info("test Dice %.4f  Jaccard %.4f", result.dice, result.jaccard)
    return model, result
