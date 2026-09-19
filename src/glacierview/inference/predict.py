"""Run the model over a glacier's time series and turn masks into areas.

This is the shared core of both inference entry points. `infer` renders one
glacier; `areas` batches the whole landing zone. They used to be separate
scripts that had drifted into computing subtly different things — the
smoothing happened at a different point and one padded its input with a
duplicated channel — so the sequence lives here once.

Order of operations, matching the published method:

    sort by date -> resize to 128x128 -> smooth along time
    -> derive NDSI/NDWI -> model -> resize back -> count pixels
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from skimage.filters import gaussian
from torch.utils.data import DataLoader, TensorDataset

from glacierview import config
from glacierview.log import get_logger
from glacierview.models import IN_CHANNELS
from glacierview.preprocess import add_spectral_indices, prepare_glacier_stack

logger = get_logger(__name__)

#: How an area is derived from the probability map. Three variants are kept
#: because the published analysis compared them.
AREA_VARIANTS: dict[str, callable] = {
    "raw": lambda p: p.clamp(min=0),
    "thresholded": lambda p: torch.where(
        p > config.PROB_THRESHOLD, p, torch.zeros_like(p)
    ),
    "binary": lambda p: (p > config.PROB_THRESHOLD).double(),
}


def dates_from_filenames(file_names: list[str]) -> list[datetime]:
    """Parse the acquisition date out of each filename.

    Filenames are load-bearing: ``{glims_id}_{date}_{L5|L7|L8}_...``, so token
    1 is the date.
    """
    return [datetime.strptime(f.split("_")[1], "%Y-%m-%d") for f in file_names]


def build_inputs(
    glacier_dir: Path | str,
    dem_path: Path | str,
    filtered_file_path: Path | str | None = None,
) -> tuple[torch.Tensor, np.ndarray, list[str], list[tuple[int, int]]]:
    """Read one glacier and return a model-ready batch.

    Returns:
        inputs: (N, IN_CHANNELS, 128, 128) float tensor.
        smoothed: the pre-index stack, kept for RGB rendering.
        file_names: source filenames in date order.
        original_sizes: each scene's native (height, width), for area.
    """
    stack, file_names, original_sizes = prepare_glacier_stack(
        str(glacier_dir), str(dem_path), filtered_file_path=filtered_file_path
    )
    # A scene with no valid pixels normalises to NaN; fill from the rest of
    # the stack rather than letting it reach the model.
    stack = np.nan_to_num(stack, nan=float(np.nanmean(stack)))

    # Smooth along time only — each pixel against itself on neighbouring
    # dates. Not a spatial blur.
    smoothed = gaussian(
        stack, sigma=[config.TIME_SMOOTHING_SIGMA, 0, 0, 0], mode="reflect"
    )
    inputs = add_spectral_indices(torch.tensor(smoothed).permute(0, 3, 1, 2))
    if inputs.shape[1] != IN_CHANNELS:
        raise ValueError(
            f"built {inputs.shape[1]} channels, model expects {IN_CHANNELS}"
        )
    return inputs, smoothed, file_names, original_sizes


def predict_probabilities(
    model, inputs: torch.Tensor, device: torch.device, batch_size: int = 64
) -> torch.Tensor:
    """Per-pixel glacier probability, as (N, 1, H, W) on the CPU."""
    batches = DataLoader(TensorDataset(inputs), batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (batch,) in batches:
            logits = model(batch.to(device=device, dtype=torch.float))
            out.append(torch.softmax(logits, dim=1)[:, 1].unsqueeze(1).cpu())
    return torch.cat(out, dim=0)


def areas_km2(
    probabilities: torch.Tensor, original_sizes: list[tuple[int, int]]
) -> list[float]:
    """Resize each prediction to its source resolution and sum its area.

    Areas must be measured natively: the model works on 128x128 crops, but a
    pixel only represents 900 m^2 in the original raster.
    """
    out = []
    for prediction, size in zip(probabilities, original_sizes, strict=True):
        resized = torchvision.transforms.Resize(size, antialias=True)(prediction)
        out.append(float(resized.sum()) * config.KM2_PER_PIXEL)
    return out


def segment_glacier(
    model,
    glims_id: str,
    device: torch.device,
    data_label: str | None = None,
    variants: tuple[str, ...] = ("binary",),
) -> dict:
    """Segment one glacier end to end.

    Returns a dict with ``file_names``, ``dates``, ``probabilities``,
    ``smoothed`` and an ``areas`` mapping of variant name to km^2 series.
    """
    inputs, smoothed, file_names, sizes = build_inputs(
        config.glacier_dir(glims_id, data_label),
        config.dem_path(glims_id, data_label),
    )
    logger.info("%s: %d scenes", glims_id, len(file_names))

    probabilities = predict_probabilities(model, inputs, device)
    areas = {
        name: areas_km2(AREA_VARIANTS[name](probabilities), sizes) for name in variants
    }
    return {
        "glims_id": glims_id,
        "file_names": file_names,
        "dates": dates_from_filenames(file_names),
        "probabilities": probabilities,
        "smoothed": smoothed,
        "areas": areas,
    }
