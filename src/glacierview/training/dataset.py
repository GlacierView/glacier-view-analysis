"""The training dataset and the manifest that selects it."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from skimage import io
from torch.utils.data import Dataset

from glacierview import config
from glacierview.log import get_logger
from glacierview.models import IN_CHANNELS

logger = get_logger(__name__)

#: A mask with less variance than this is effectively blank — the glacier is
#: missing or the outline failed to rasterise — so the pair is dropped.
MIN_MASK_VARIANCE = 0.001
#: Keeps a uniform channel from dividing by zero during normalisation.
NORMALIZE_SMOOTHING = 1


def build_manifest(
    data_dir: Path, test_size: float = 0.1, seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair images with masks, drop blank masks, and split.

    Images and masks are matched on **GLIMS ID** — the first
    underscore-delimited field of the filename — not on the whole filename.
    The two directories are named differently in the real dataset:

        images/G007026E45991N_2009-09-30_L5_C02_T1_L2_SR.tif
        masks/G007026E45991N.tif

    One mask per glacier serves every scene of that glacier, since the outline
    is a single point-in-time annotation. Matching on ID also covers the
    simpler layout where both sides share a name.

    Args:
        data_dir: holds ``images/`` and ``masks/``.
        test_size: fraction held out.
        seed: set for a reproducible split. The original script shuffled
            unseeded, so no two runs were comparable.

    Returns:
        (train, holdout) frames of relative ``img`` / ``mask`` paths.
    """
    images_dir, masks_dir = data_dir / "images", data_dir / "masks"
    for d in (images_dir, masks_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"expected {d}")

    def glims_id(name: str) -> str:
        # Strip the suffix first: a mask is named "G007026E45991N.tif" with no
        # underscore at all, so splitting before stripping keeps ".tif".
        return Path(name).stem.split("_")[0]

    # One mask per glacier; keep only the ones carrying an actual outline.
    masks, blank = {}, 0
    for path in sorted(masks_dir.iterdir()):
        if path.suffix != ".tif" or path.name.startswith("."):
            continue
        if io.imread(path).var() > MIN_MASK_VARIANCE:
            masks[glims_id(path.name)] = path.name
        else:
            blank += 1

    rows, unmatched = [], 0
    for path in sorted(images_dir.iterdir()):
        if path.suffix != ".tif" or path.name.startswith("."):
            continue
        mask_name = masks.get(glims_id(path.name))
        if mask_name is None:
            unmatched += 1
            continue
        rows.append({"img": f"images/{path.name}", "mask": f"masks/{mask_name}"})

    logger.info(
        "%d pairs from %d usable masks (%d blank, %d images without a mask)",
        len(rows), len(masks), blank, unmatched,
    )
    if not rows:
        raise RuntimeError(
            f"no image/mask pairs in {data_dir}; images and masks are matched "
            f"on the GLIMS ID prefix of their filenames"
        )

    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int((1 - test_size) * len(frame))
    return frame[:split].reset_index(drop=True), frame[split:].reset_index(drop=True)


def _to_chw(array: np.ndarray) -> np.ndarray:
    """Normalise a tif's array to channel-first (C, H, W).

    Two vintages of training tif exist on disk and they are not shaped the
    same: the current set reads back as ``(1, 128, 128, 7)`` while the older
    one is ``(128, 128, 8)``. Masks come as ``(128, 128, 1)``. Rather than
    assume, squeeze away leading singleton axes and then move the smallest
    trailing axis to the front.
    """
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D raster, got shape {array.shape}")
    # Channel axis is whichever is smallest; for 128x128xC that is the last.
    if array.shape[-1] <= array.shape[0]:
        return np.transpose(array, (2, 0, 1))
    return array


class GlacierDataset(Dataset):
    """Image/mask pairs, normalised, with NDSI and NDWI prepended.

    The tifs on disk carry 7 channels (6 Landsat bands + DEM); the two
    spectral indices bring that to the 9 the model takes.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        data_dir: Path,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.manifest)

    def _augment(self, image, label):
        """Flip and blur, each independently at p=0.5.

        Applying all three every time would quadruple the data; independent
        coin flips grow it ~75% while still covering the combinations.
        """
        if random.random() > 0.5:
            image, label = TF.vflip(image), TF.vflip(label)
        if random.random() > 0.5:
            image, label = TF.hflip(image), TF.hflip(label)
        if random.random() > 0.5:
            image, label = TF.gaussian_blur(image, 3), TF.gaussian_blur(label, 3)
        return image, label

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img_path = self.data_dir / row["img"]
        image = torch.tensor(_to_chw(io.imread(img_path))).float()

        if self.normalize:
            # Per-channel min-max onto [0, 1], driven by the image's own
            # channel count rather than a hardcoded width.
            mins = [image[c].min() for c in range(image.shape[0])]
            spans = [
                image[c].max() - image[c].min() + NORMALIZE_SMOOTHING
                for c in range(image.shape[0])
            ]
            image = TF.normalize(image, mins, spans)

        # Band positions refer to config.COMMON_BANDS order.
        green, nir, swir = image[1], image[3], image[4]
        eps = config.INDEX_EPSILON
        ndsi = ((green - swir + eps) / (green + swir + eps)).unsqueeze(0)
        image = torch.cat((ndsi, image), dim=0)
        ndwi = ((green - nir + eps) / (green + nir + eps)).unsqueeze(0)
        image = torch.cat((ndwi, image), dim=0)

        if image.shape[0] != IN_CHANNELS:
            raise ValueError(
                f"{img_path}: built {image.shape[0]} channels, "
                f"model expects {IN_CHANNELS}"
            )

        label = torch.tensor(_to_chw(io.imread(self.data_dir / row["mask"])))
        if self.augment:
            image, label = self._augment(image, label)
        return image, label
