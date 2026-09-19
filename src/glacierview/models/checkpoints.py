"""Load and save model checkpoints.

Checkpoints in this project are ``torch.save(model)`` output — a pickled
module, not a state dict. That has one awkward consequence worth isolating in
one place: the pickle records where the class came from, and the released
checkpoints were written by a script, so they reference ``__main__.UNet`` and
``__main__.conv_block``. Loading one from anywhere that is not that original
script fails with::

    AttributeError: Can't get attribute 'UNet' on <module '__main__'>

`load_checkpoint` injects the names into ``__main__`` before unpickling, which
is why callers no longer need to import the classes for their side effect.

Prefer `save_state_dict` for anything shared outside this repo: a state dict
carries no class references and no arbitrary-code-execution risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from glacierview.log import get_logger
from glacierview.models.unet import IN_CHANNELS, UNet, conv_block

logger = get_logger(__name__)


def _inject_legacy_names() -> None:
    """Make ``__main__.UNet`` / ``__main__.conv_block`` resolvable."""
    main = sys.modules["__main__"]
    for name, obj in (("UNet", UNet), ("conv_block", conv_block)):
        if not hasattr(main, name):
            setattr(main, name, obj)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> UNet:
    """Load a pickled-module checkpoint and put it in eval mode.

    Raises:
        FileNotFoundError: with the searched path, rather than torch's less
            helpful error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}. Released checkpoints live in "
            f"s3://segmentation-model-gv/checkpoints/pytorch/ — see "
            f"docs/MODEL_MANIFEST.md"
        )
    _inject_legacy_names()
    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device).eval()

    channels = model.resnet[0].weight.shape[1]
    if channels != IN_CHANNELS:
        logger.warning(
            "checkpoint %s takes %d input channels but this code builds %d",
            path.name, channels, IN_CHANNELS,
        )
    logger.info("loaded %s (%d input channels) onto %s", path.name, channels, device)
    return model


def save_state_dict(model: UNet, path: str | Path) -> Path:
    """Write weights only — portable, and safe to share."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("wrote state dict to %s", path)
    return path
