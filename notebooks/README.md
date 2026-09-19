# Notebooks

Notebooks are for work that genuinely benefits from cell-by-cell iteration.
Anything that runs the same way every time lives in the `glacierview` package
and is driven from the CLI instead — see `glacierview --help`.

All of these import the installed package, so run `uv sync` first and start
Jupyter from anywhere:

```bash
uv run jupyter lab
```

None of them manipulate `sys.path` or hardcode a home directory any more;
paths come from `glacierview.config`, which is environment-overridable.

## `pipeline/`

Numbered because order matters — these produce the datasets everything else
consumes. They are kept as notebooks because each step is run rarely, is
inspected as it goes, and involves judgement calls about what to keep.

| Notebook | What it does |
|---|---|
| `01_glims_select_glaciers.ipynb` | Filters the GLIMS inventory to 18,093 glaciers and computes the bounding boxes Earth Engine is queried with. |
| `02_glims_to_parquet.ipynb` | Adds `geog_area_rollup`, drops geometry, writes the Parquet that Athena reads. |
| `03_ee_pull_training.ipynb` | Downloads one temporal composite per glacier for training. |
| `04_ee_pull_dems.ipynb` | Downloads one NASADEM per glacier. |
| `05_ee_pull_time_series.ipynb` | Downloads the full 1984-present series for the inference set. |
| `06_metadata_training.ipynb` | Relationalises the training pull's logs and metadata into CSVs. |
| `07_metadata_time_series.ipynb` | The same for the inference pull. These CSVs become the Athena tables. |
| `08_inventory_downloads.ipynb` | Inventories what actually landed on disk. |
| `09_build_masks.ipynb` | Rasterises GLIMS polygons into masks, reprojecting per image CRS. |
| `10_build_training_set.ipynb` | Pairs images with masks into the `images/` + `masks/` training set. |

The Earth Engine steps use `glacierview.earthengine.EePull`; the raster steps
use `glacierview.rasters` and `glacierview.preprocess`. The notebooks are
drivers, not implementations.

## `exploration/`

Sandboxes. Not expected to run top to bottom, and some depend on cells that
were edited away — that is what they are for.

| Notebook | What it does |
|---|---|
| `glims_inventory.ipynb` | Region and size distributions across GLIMS. |
| `landsat_metadata.ipynb` | Where the cloud-cover and zero-pixel thresholds came from. |
| `band_distributions.ipynb` | Per-band pixel-value distributions. |
| `preprocessing_sandbox.ipynb` | Scratch space for the preprocessing functions. |
| `inspect_training_data.ipynb` | Visual QA of the built training set. Run this first to see what the data looks like. |
| `reprojection.ipynb` | CRS reprojection experiments. |

## `analysis/`

| Notebook | What it does |
|---|---|
| `areas_true_vs_predicted.ipynb` | Compares predicted areas against GLIMS `db_area` by region, via log-linear regression and a bootstrap. Reads the committed CSVs in `data/analysis/`. |

## Not here any more

`glacier_segmentation-updated.ipynb`, `test_saved_model_matt.ipynb` and
`test_saved_model_somansh.ipynb` were removed: training is now
`glacierview train` and single-glacier inference is `glacierview infer`.
`get_s3_data.ipynb` did what `aws s3 cp` does. See git history.
