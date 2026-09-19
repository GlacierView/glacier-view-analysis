"""Training: dataset construction, losses, and the training loop."""

from glacierview.training.dataset import GlacierDataset, build_manifest
from glacierview.training.trainer import TrainConfig, TrainResult, fit

__all__ = ["GlacierDataset", "TrainConfig", "TrainResult", "build_manifest", "fit"]
