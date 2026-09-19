"""Segment every glacier in the landing zone and write surface-area time series.

This is the batch counterpart to inference/infer.py; both share the
preprocessing pipeline and the model, so their numbers should agree.

Run from src/segmentation:

    python final_areas.py

Outputs, all relative to the working directory:

    areas_no_threshold.csv   annual mean area per glacier, raw probabilities
    areas_05_thresh.csv      ... keeping probabilities above PROB_THRESH
    areas_binary_05.csv      ... counting thresholded pixels as 1
    glacier_areas/<id>_areas.csv   per-date series for one glacier
    gifs/<id>_final.gif      the prediction animated over the series

The three aggregate CSVs are rewritten after every glacier, so a long run that
dies partway still leaves usable output.
"""

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from skimage.filters import gaussian
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SEGMENTATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SEGMENTATION_DIR))

from helpers.preprocess import add_spectral_indices, prepare_glacier_stack  # noqa: E402
from inference.cnn import IN_CHANNELS, UNet, conv_block  # noqa: E402,F401

# UNet and conv_block are imported into this module's globals on purpose: the
# released checkpoints are pickled modules referencing `__main__.UNet` and
# `__main__.conv_block`, so both names must resolve here for torch.load.

#: Probability above which a pixel counts as glacier.
PROB_THRESH = 0.5
#: A Landsat pixel is 30 m x 30 m = 900 m^2 = 0.0009 km^2.
KM2_PER_PIXEL = 0.0009
#: Std. dev. of the Gaussian filter applied along the *time* axis, in frames.
TIME_SMOOTHING_SIGMA = 20
#: Which landing-zone subdirectory to read.
DATA_LABEL = "full_time_series_c02_t1_l2"
#: Monthly index the annual means are merged onto.
DATE_INDEX = pd.date_range(start="1979-01-01", end="2024-01-01", freq="MS")
#: Keep every Nth frame in the GIF.
GIF_FRAME_STRIDE = 10
#: Checkpoint path, relative to the working directory.
MODEL_PATH = "model"

#: The three ways an area is derived from the model's probability map. Each
#: produces its own CSV so they can be compared downstream.
AREA_VARIANTS = (
    ("areas_no_threshold.csv", lambda p: p.clamp(min=0)),
    ("areas_05_thresh.csv", lambda p: torch.where(p > PROB_THRESH, p, torch.zeros_like(p))),
    ("areas_binary_05.csv", lambda p: (p > PROB_THRESH).double()),
)


def predict_probabilities(model, inputs, device):
    """Return the model's per-pixel glacier probability as (N, 1, H, W)."""
    batches = DataLoader(TensorDataset(inputs), batch_size=64, shuffle=False)
    out = []
    with torch.no_grad():
        for (batch,) in tqdm(batches, leave=False):
            logits = model(batch.to(device=device, dtype=torch.float))
            out.append(torch.softmax(logits, dim=1)[:, 1].unsqueeze(1).cpu())
    return torch.cat(out, dim=0)


def areas_km2(probabilities, original_sizes):
    """Resize each prediction back to its source resolution and sum its area.

    Areas must be measured at native resolution: the model works on 128x128
    crops, but a pixel only represents 900 m^2 in the original raster.
    """
    areas = []
    for prediction, size in zip(probabilities, original_sizes):
        resized = torchvision.transforms.Resize(size, antialias=True)(prediction)
        areas.append(float(resized.sum()) * KM2_PER_PIXEL)
    return areas


def to_annual_means(areas, dates, glims_id):
    """Collapse a per-date series to one value per year, stamped mid-year.

    The mid-year offset makes the annual value plot at the centre of the period
    it summarises rather than at January.
    """
    series = pd.DataFrame({glims_id: areas}, index=pd.DatetimeIndex(dates))
    annual = series.groupby(pd.PeriodIndex(series.index, freq="Y"))[glims_id].mean()
    annual.index = annual.index.to_timestamp() + pd.DateOffset(months=6)
    return annual.to_frame().rename_axis("Dates").reset_index()


def write_gif(gif_path, scratch_dir, images, probabilities, file_names):
    """Animate the prediction over the series, overlaid on the RGB composite."""
    os.makedirs(scratch_dir, exist_ok=True)
    for stale in os.listdir(scratch_dir):
        os.remove(os.path.join(scratch_dir, stale))

    masks = probabilities.numpy()
    for i in range(0, len(file_names), GIF_FRAME_STRIDE):
        rgb = images[i][:, :, [2, 1, 0]]
        fig, axs = plt.subplots(2, figsize=(10, 10))
        fig.suptitle(file_names[i])
        axs[0].imshow(rgb)
        axs[1].imshow(rgb)
        axs[1].imshow(masks[i, 0], alpha=0.2)
        fig.savefig(os.path.join(scratch_dir, f"{file_names[i]}_final.png"), dpi=100)
        plt.close(fig)

    with imageio.get_writer(gif_path, mode='I') as writer:
        for frame in sorted(os.listdir(scratch_dir)):
            writer.append_data(imageio.imread(os.path.join(scratch_dir, frame)))


def process_glacier(glims_id, landsat_dir, dem_dir, model, device):
    """Run one glacier end to end. Returns {csv_name: annual-mean frame}."""
    stack, file_names, original_sizes = prepare_glacier_stack(
        glacier_dir=os.path.join(landsat_dir, glims_id),
        dem_path=os.path.join(dem_dir, f"{glims_id}_NASADEM.tif"),
    )
    # A scene with no valid pixels normalizes to NaN; fill from the rest of
    # the stack rather than propagating it into the model.
    stack = np.nan_to_num(stack, nan=float(np.nanmean(stack)))

    dates = [datetime.strptime(f.split("_")[1], "%Y-%m-%d") for f in file_names]

    # Smooth along time only, then derive the indices from the smoothed bands
    # (this order matches inference/infer.py and the published method).
    smoothed = gaussian(stack, sigma=[TIME_SMOOTHING_SIGMA, 0, 0, 0], mode="reflect")
    inputs = add_spectral_indices(torch.tensor(smoothed).permute(0, 3, 1, 2))
    if inputs.shape[1] != IN_CHANNELS:
        raise ValueError(
            f"built {inputs.shape[1]} channels, model expects {IN_CHANNELS}"
        )

    probabilities = predict_probabilities(model, inputs, device)

    frames = {}
    for csv_name, to_mask in AREA_VARIANTS:
        areas = areas_km2(to_mask(probabilities), original_sizes)
        frames[csv_name] = to_annual_means(areas, dates, glims_id)
        if csv_name == "areas_no_threshold.csv":
            per_date = pd.DataFrame({glims_id: areas}, index=pd.DatetimeIndex(dates))
            per_date.rename_axis("Dates").to_csv(
                os.path.join("glacier_areas", f"{glims_id}_areas.csv")
            )

    write_gif(
        gif_path=os.path.join("gifs", f"{glims_id}_final.gif"),
        scratch_dir=str(SEGMENTATION_DIR / "tmp" / glims_id),
        images=smoothed,
        probabilities=probabilities,
        file_names=file_names,
    )
    return frames


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device).eval()

    landing_zone = SEGMENTATION_DIR.parent / "earth_engine" / "data" / "ee_landing_zone" / DATA_LABEL
    landsat_dir = str(landing_zone / "landsat")
    dem_dir = str(landing_zone / "dems")

    os.makedirs("glacier_areas", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)

    glacier_ids = sorted(g for g in os.listdir(landsat_dir) if not g.startswith("."))
    # One accumulating wide frame per variant: rows are dates, columns glaciers.
    aggregates = {name: pd.DataFrame({"Dates": DATE_INDEX}) for name, _ in AREA_VARIANTS}

    failures = []
    for i, glims_id in enumerate(glacier_ids, start=1):
        try:
            frames = process_glacier(glims_id, landsat_dir, dem_dir, model, device)
        except Exception:
            failures.append(glims_id)
            print(f"[{i}/{len(glacier_ids)}] FAILED {glims_id}", file=sys.stderr)
            traceback.print_exc()
            continue

        for name, frame in frames.items():
            aggregates[name] = aggregates[name].merge(frame, how="left", on="Dates")
            # Rewritten every glacier so a crash still leaves usable output.
            aggregates[name].to_csv(name, index=False)
        print(f"[{i}/{len(glacier_ids)}] {glims_id}")

    print(f"\n{len(glacier_ids) - len(failures)}/{len(glacier_ids)} glaciers succeeded")
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)


if __name__ == "__main__":
    main()
