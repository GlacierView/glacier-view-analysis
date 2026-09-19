"""Turn raw Landsat rasters into model-ready input.

Both inference entry points (`inference/infer.py` and `final_areas.py`) share
this module, so a change here affects them equally — which is the point. The
pipeline is:

    read rasters -> select common bands -> normalize -> resize to 128x128
    -> stack the DEM on -> smooth along time -> prepend NDSI and NDWI

`prepare_glacier_stack` covers everything up to the DEM stack;
`add_spectral_indices` does the last step. Time smoothing is left to the
callers because it operates across a whole series at once.
"""

import numpy as np
import torch
import torchvision
from helpers import read
from helpers.landsat_bands import landsat_bands

#: Bands taken from every scene, in the order the model expects. `swir_2` is
#: excluded deliberately — it was dropped for quality. See inference/cnn.py.
COMMON_BANDS = ['blue', 'green', 'red', 'nir', 'swir', 'thermal']
#: The model's fixed input resolution.
MODEL_INPUT_DIM = (128, 128)
#: Added to both sides of the index ratios to avoid dividing by zero.
INDEX_EPSILON = 1e-4


def get_common_bands(rasters, common_bands: list):
    """Keep only `common_bands`, so every satellite yields the same channels.

    Landsat 5/7 and Landsat 8 order their bands differently (L8 has an extra
    coastal-aerosol band first), so the band index for e.g. `swir` depends on
    the satellite. The satellite is read from the third underscore-delimited
    field of the filename and looked up in `landsat_bands`.

    Note the output keeps each satellite's *native* band order filtered down,
    not the order of `common_bands` — which is what makes the result
    consistent across satellites.
    """
    selected = {}
    for file_name, raster in rasters.items():
        satellite = file_name.split("_")[2].lower()
        band_names = list(landsat_bands[satellite].keys())
        keep = np.isin(band_names, common_bands)
        selected[file_name] = raster[:, :, keep]
    return selected


def normalize_rasters(rasters):
    """Scale each band to roughly [0, 1], ignoring single-pixel extremes.

    This is deliberately not a plain min-max. The true minimum and maximum of a
    Landsat band are usually artifacts — dead or saturated pixels — and scaling
    to them crushes the real signal into a narrow range. So the *second*
    smallest and second largest distinct values are used as the bounds, and the
    true extremes are clamped to them first.
    """
    normalized = {}
    for file_name, raster in rasters.items():
        out = raster.astype(np.double, copy=True)
        for i in range(out.shape[2]):
            band = out[:, :, i]
            low, high = band.min(), band.max()

            if low == high:
                # Uniform band: nothing to stretch. Scale to 1 if non-zero.
                if low != 0:
                    out[:, :, i] = band / low
                continue

            distinct = np.unique(band)
            if distinct.size < 4:
                # Too few values for the second-extreme trick; fall back.
                second_low, second_high = low, high
            else:
                second_low, second_high = distinct[1], distinct[-2]

            clamped = np.clip(band, second_low, second_high)
            out[:, :, i] = (clamped - second_low) / (second_high - second_low)
        normalized[file_name] = out
    return normalized


def resize_rasters(rasters, dim):
    """Resize every raster to `dim`, preserving the HWC layout."""
    resizer = torchvision.transforms.Resize(dim)
    resized = {}
    for file_name, raster in rasters.items():
        chw = torch.from_numpy(raster).permute(2, 0, 1)
        resized[file_name] = resizer(chw).permute(1, 2, 0).numpy()
    return resized


def prepare_glacier_stack(glacier_dir, dem_path, filtered_file_path=None):
    """Read one glacier's scenes and return a model-ready stack.

    Returns:
        stack: (N, 128, 128, len(COMMON_BANDS) + 1), date-ordered, DEM last.
        file_names: the source filenames in the same order.
        original_sizes: each source image's (height, width), so predictions can
            be resized back before areas are measured.
    """
    images = read.get_rasters(glacier_dir, filtered_file_path=filtered_file_path)
    if not images:
        raise FileNotFoundError(f"no .tif files found in {glacier_dir}")

    file_names = sorted(images)
    original_sizes = [images[f].shape[:2] for f in file_names]

    images = get_common_bands(images, COMMON_BANDS)
    images = normalize_rasters(images)
    images = resize_rasters(images, MODEL_INPUT_DIM)

    # One static DEM per glacier, reused for every date. It is resized before
    # being normalized while the scenes are normalized before being resized;
    # that asymmetry is inherited from the original pipeline and kept so areas
    # stay comparable with previously published runs.
    dem = read.get_dem(dem_path)
    dem_key = next(iter(dem))
    dem = resize_rasters(dem, MODEL_INPUT_DIM)
    dem = normalize_rasters(dem)

    stack = np.stack([
        np.concatenate((images[f], dem[dem_key]), axis=2) for f in file_names
    ])
    return stack, file_names, original_sizes


def add_spectral_indices(inputs):
    """Prepend NDSI then NDWI to an (N, C, H, W) batch.

    Each is prepended in turn, so NDWI ends up at channel 0 and NDSI at 1 —
    the order the model was trained on. Band positions refer to COMMON_BANDS
    (green=1, nir=3, swir=4) as they are *before* any prepending.

    NDSI (snow) is high for ice: bright in green, dark in shortwave infrared.
    NDWI (water) is high for meltwater, a common false positive.
    """
    eps = INDEX_EPSILON
    green, nir, swir = inputs[:, 1], inputs[:, 3], inputs[:, 4]

    ndsi = ((green - swir + eps) / (green + swir + eps)).unsqueeze(1)
    inputs = torch.cat((ndsi, inputs), dim=1)

    ndwi = ((green - nir + eps) / (green + nir + eps)).unsqueeze(1)
    inputs = torch.cat((ndwi, inputs), dim=1)
    return inputs
