"""Paths and pipeline constants, resolved once here instead of per script.

Every path is derived from two roots, both overridable by environment variable
so nothing is hardcoded to one machine:

``GV_REPO_ROOT``
    The checkout. Defaults to the repository this file lives in.

``GV_DATA_ROOT``
    Where the imagery, DEMs and metadata live. Defaults to the historical
    in-repo location (``<repo>/src``), because that tree holds ~110 GB that
    predates this layout. Point it at an external disk to work off one:

        export GV_DATA_ROOT=/Volumes/T7/GlacierView

``data_label`` selects which dataset under the landing zone to use. Two are
live; the others are pre-Collection-2 and should not be populated.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Roots
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(os.environ.get("GV_REPO_ROOT", Path(__file__).resolve().parents[2]))
DATA_ROOT = Path(os.environ.get("GV_DATA_ROOT", REPO_ROOT / "src"))

#: Committed reference data for the analysis notebooks.
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"
#: A 32-pair training fixture, committed so training can be smoke tested
#: without the full set.
SAMPLE_TRAINING_DIR = REPO_ROOT / "data" / "sample" / "training"
#: Athena queries, run by hand against the glacier-view catalog.
SQL_DIR = REPO_ROOT / "sql"

# --------------------------------------------------------------------------- #
# Dataset selection
# --------------------------------------------------------------------------- #

#: Full inference time series, 1984-present. Collection 2, Tier 1, Level 2.
INFERENCE_DATA_LABEL = "full_time_series_c02_t1_l2"
#: One temporal composite per glacier, used for training.
TRAINING_DATA_LABEL = "localized_time_series_for_training_c02_t1_l2"

DEFAULT_DATA_LABEL = os.environ.get("GV_DATA_LABEL", INFERENCE_DATA_LABEL)

# --------------------------------------------------------------------------- #
# Model and preprocessing constants
# --------------------------------------------------------------------------- #

#: The model's fixed input resolution.
MODEL_INPUT_DIM = (128, 128)
#: Probability above which a pixel counts as glacier.
PROB_THRESHOLD = 0.5
#: A Landsat pixel is 30 m x 30 m = 900 m^2 = 0.0009 km^2.
KM2_PER_PIXEL = 0.0009
#: Std. dev. of the Gaussian filter applied along the *time* axis, in frames.
TIME_SMOOTHING_SIGMA = 20
#: Bands taken from every scene. `swir_2` is excluded deliberately — it was
#: dropped for quality; see glacierview.models.unet.
COMMON_BANDS = ("blue", "green", "red", "nir", "swir", "thermal")
#: Added to both sides of the index ratios to avoid dividing by zero.
INDEX_EPSILON = 1e-4


# --------------------------------------------------------------------------- #
# Derived paths
# --------------------------------------------------------------------------- #

def landing_zone(data_label: str | None = None) -> Path:
    """Root of one dataset's downloaded imagery, DEMs and logs."""
    label = data_label or DEFAULT_DATA_LABEL
    return DATA_ROOT / "earth_engine" / "data" / "ee_landing_zone" / label


def landsat_dir(data_label: str | None = None) -> Path:
    """Per-glacier scene directories: ``landsat/<glims_id>/*.tif``."""
    return landing_zone(data_label) / "landsat"


def glacier_dir(glims_id: str, data_label: str | None = None) -> Path:
    """One glacier's scenes."""
    return landsat_dir(data_label) / glims_id


def dem_path(glims_id: str, data_label: str | None = None) -> Path:
    """One glacier's NASADEM. Static: reused for every date."""
    return landing_zone(data_label) / "dems" / f"{glims_id}_NASADEM.tif"


def log_dir(data_label: str | None = None) -> Path:
    """Earth Engine download logs for one dataset."""
    return landing_zone(data_label) / "logs"


def processed_metadata_dir(data_label: str | None = None) -> Path:
    """Relationalised metadata CSVs, the basis of the Athena tables."""
    label = data_label or DEFAULT_DATA_LABEL
    return DATA_ROOT / "earth_engine" / "data" / "processed_metadata" / label


def glims_data_dir() -> Path:
    """GLIMS shapefiles and the derived sample CSVs."""
    return DATA_ROOT / "glims" / "data"


def training_data_dir() -> Path:
    """Masks and the assembled image/mask pairs."""
    return DATA_ROOT / "segmentation" / "training" / "data"


def outputs_dir() -> Path:
    """Where inference writes GIFs, area CSVs and scratch frames."""
    return Path(os.environ.get("GV_OUTPUT_DIR", DATA_ROOT / "segmentation"))


def mpl_style() -> Path:
    """The shared matplotlib style, shipped as package data."""
    return Path(__file__).resolve().parent / "styles" / "ieee.mplstyle"
