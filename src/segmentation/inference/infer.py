"""Segment one glacier's Landsat time series and render a GIF plus an area plot.

Usage (from the repo's src/segmentation directory, or anywhere — the import
path is bootstrapped below):

    python inference/infer.py --glimsid G007026E45991N

For every glacier in the landing zone at once, use ../final_areas.py instead.
Both share the preprocessing helpers and the model, so they should agree.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torchvision
from skimage.filters import gaussian
from torch.utils.data import DataLoader, TensorDataset

# Make `helpers` and `inference` importable no matter which directory this is
# launched from.
SEGMENTATION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SEGMENTATION_DIR))

from helpers.preprocess import (  # noqa: E402
    add_spectral_indices, prepare_glacier_stack,
)
from inference.cnn import IN_CHANNELS, UNet, conv_block  # noqa: E402,F401

# UNet and conv_block are imported into this module's globals on purpose: the
# released checkpoints are pickled modules referencing `__main__.UNet` and
# `__main__.conv_block`, so both names must resolve here for torch.load to work.

#: Probability above which a pixel counts as glacier.
PROB_THRESH = 0.5
#: A Landsat pixel is 30 m x 30 m = 900 m^2 = 0.0009 km^2.
KM2_PER_PIXEL = 0.0009
#: Std. dev. of the Gaussian filter applied along the *time* axis, in frames.
TIME_SMOOTHING_SIGMA = 20
#: Keep every Nth frame in the GIF; the full series is too long to animate.
GIF_FRAME_STRIDE = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glimsid", type=str, default="G007026E45991N",
                        help="GLIMS ID of the glacier to segment.")
    parser.add_argument("--data-label", type=str, default="full_time_series_c02_t1_l2",
                        help="Landing-zone subdirectory holding the imagery.")
    parser.add_argument("--model", type=str, default="model",
                        help="Path to the checkpoint (a torch.save'd module).")
    return parser.parse_args()






def predict(model, inputs, device):
    """Return per-pixel glacier probabilities, zeroed below PROB_THRESH."""
    threshold = torch.nn.Threshold(PROB_THRESH, 0)
    batches = DataLoader(TensorDataset(inputs), batch_size=64)

    predictions = []
    with torch.no_grad():
        for (batch,) in batches:
            logits = model(batch.to(device=device, dtype=torch.float))
            glacier_prob = torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)
            predictions.append(threshold(glacier_prob).cpu())
    return torch.cat(predictions, dim=0).numpy()


def measure_areas(predictions, original_sizes):
    """Convert predicted masks to km^2 at each image's native resolution."""
    areas = []
    for prediction, size in zip(predictions, original_sizes):
        mask = torch.from_numpy(prediction)
        resized = torchvision.transforms.Resize(size, antialias=True)(mask)
        binary = (resized > PROB_THRESH).sum().item()
        areas.append(binary * KM2_PER_PIXEL)
    return areas


def write_gif(gif_path, scratch_dir, images, predictions, file_names):
    """Render every GIF_FRAME_STRIDE'th frame, then animate them in order."""
    os.makedirs(scratch_dir, exist_ok=True)
    for stale in os.listdir(scratch_dir):
        os.remove(os.path.join(scratch_dir, stale))

    for i in range(0, len(file_names), GIF_FRAME_STRIDE):
        fig, axs = plt.subplots(2, figsize=(10, 10))
        fig.suptitle(file_names[i])
        axs[0].imshow(images[i][:, :, [2, 1, 0]])   # red, green, blue
        axs[1].imshow(predictions[i, 0])
        fig.savefig(os.path.join(scratch_dir, f"{file_names[i]}.png"), dpi=100)
        plt.close(fig)

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    with imageio.get_writer(gif_path, mode='I') as writer:
        for frame in sorted(os.listdir(scratch_dir)):
            writer.append_data(imageio.imread(os.path.join(scratch_dir, frame)))


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    landing_zone = SEGMENTATION_DIR.parent / "earth_engine" / "data" / "ee_landing_zone" / args.data_label
    glacier_dir = landing_zone / "landsat" / args.glimsid
    dem_path = landing_zone / "dems" / f"{args.glimsid}_NASADEM.tif"

    stack, file_names, original_sizes = prepare_glacier_stack(
        str(glacier_dir), str(dem_path)
    )
    dates = [datetime.strptime(f.split("_")[1], '%Y-%m-%d') for f in file_names]

    # Smooth along the time axis only, so each pixel is averaged against
    # itself on neighbouring dates. This runs before the indices are derived.
    smoothed = gaussian(stack, sigma=[TIME_SMOOTHING_SIGMA, 0, 0, 0], mode='reflect')

    inputs = torch.tensor(smoothed).permute(0, 3, 1, 2)
    inputs = add_spectral_indices(inputs)
    assert inputs.shape[1] == IN_CHANNELS, (
        f"built {inputs.shape[1]} channels, model expects {IN_CHANNELS}"
    )

    model = torch.load(args.model, map_location=device).eval()
    predictions = predict(model, inputs, device)

    out_dir = SEGMENTATION_DIR / "gifs"
    write_gif(
        gif_path=str(out_dir / f"{args.glimsid}.gif"),
        scratch_dir=str(SEGMENTATION_DIR / "tmp" / "gif_creation"),
        images=smoothed,
        predictions=predictions,
        file_names=file_names,
    )

    areas = measure_areas(predictions, original_sizes)
    plt.figure()
    plt.plot(dates, areas)
    plt.title(f"{args.glimsid} estimated surface area")
    plt.ylabel("km$^2$")
    plt.savefig(str(out_dir / f"{args.glimsid}.png"))
    print(f"{len(file_names)} images; area {areas[0]:.2f} -> {areas[-1]:.2f} km^2")


if __name__ == "__main__":
    main()
