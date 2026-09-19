# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research pipeline that measures glacier surface-area change over time. It pulls Landsat 5/7/8 imagery
and NASADEM elevation from Google Earth Engine for glaciers identified by GLIMS ID, segments each image
with a ResNet-50-encoder U-Net (PyTorch), and converts the predicted masks into per-glacier area time
series (1984–present) plus GIFs.

## The paper

The work is written up as an IEEE journal submission: **"A Large Scale Analysis of Mountain Glacier
Shrinkage Using Convolutional Neural Networks"** — Matthew Waismann (Amazon), Somansh Budhwar, and
Armin Schwartzman (Halıcıoğlu Data Science Institute, UC San Diego; `armins@ucsd.edu`). The LaTeX
source is not in this repo. Headline claims:

- Trained on **7,174** glacier contours + images, one time point each; **Dice 0.92** on the test set.
- Inference over **505,835 images for 1,083 glaciers**, 1984–2023.
- Overall glacier surface-area decline of **0.2% ± 0.01% per year**, negative and significant in all
  five regions.

Per-region relative area change per year (log-linear regression):

| Region | Δ/yr | Std err |
|---|---|---|
| Europe | −0.00256 | 0.00019 |
| South America | −0.00256 | 0.00029 |
| Caucasus | −0.00167 | 0.00025 |
| Asia | −0.00163 | 0.00015 |
| North America | −0.00155 | 0.00017 |
| **Total** | **−0.00199** | **0.00010** |

Cumulative 1985–2022 declines from the bootstrap analysis: Europe 16.2%, Caucasus 9.1%, Asia 7.1%,
South America 2.6%, North America 2.3%.

The GLIMS funnel, with the paper's numbers: **328,115** glaciers in the snapshot → **170,363** after
"outline drawn after 2005" → **122,212** after excluding Arctic/Antarctic regions → **18,093** after
the 1–50 km² area filter. That last number matches the repo exactly.

The paper is a work in progress and carries `\editcom{}` notes; two of its own numbers are stale
(the conclusion says "a dataset of 10,443 glaciers", which is really the 10,447-row
`training_data_set.csv`, and its body text says 10 input channels where its channel figure says 9 —
the figure is right, see "Model contract").

`docs/low_level_design.md` is the design doc — read it before changing anything in the data extraction,
metadata, or pre-inference filtering steps, since it records *why* the filter thresholds are what they
are and what the known tech debt is. Its concrete names (Glue tables, model band order) have drifted
from the code; where they disagree, the "Data" and "Model contract" sections below were checked against
the live account and the source.

## Pipeline stages

Data flows in one direction; each stage writes files that the next stage reads.

1. **GLIMS selection** — `src/glims/src/` notebooks read the GLIMS polygon shapefile, roll `geog_area`
   up into 5 continental buckets (`geog_area_rollup`), and emit bounding boxes + sample CSVs.
2. **Earth Engine pull** — `src/earth_engine/src/gee_readers/ee_helpers.py` (`EePull` class) downloads
   one GeoTIFF per image via `geemap.ee_export_image` at 30 m scale, plus a NASADEM per glacier.
   Server-side filters: `CLOUD_COVER <= 10`, `IMAGE_QUALITY == 9` (`IMAGE_QUALITY_OLI` for L8).
   Each image's `getInfo()` JSON is appended to `meta_data/metadata_list_l{5,7,8}` alongside the tifs,
   and a log line per image goes to the run log.
3. **Metadata relationalization** — `src/earth_engine/src/metadata/time_series_metadata_handler.ipynb`
   parses the logs, dedupes the EE metadata blobs (EE sometimes returns >1 record per image), opens each
   tif with rasterio for image/file attributes, and writes CSVs → S3 → AWS Glue → Athena
   (database `glacier-view` in us-west-1 — see "Athena / Glue" below).
4. **SQL filtering** — `src/sql/` queries run against Athena to pick which GLIMS IDs and which individual
   files are eligible. `identify_inference_*` picks glaciers (N per `geog_area_rollup`);
   `inference_data_query.sql` / `training_data_query.sql` pick files. Criteria: cloud cover < 10%,
   summer months only (May–Oct northern hemisphere, Nov–Apr southern), > 50,000 pixels,
   0 no-data pixels, < 10% zero pixels.
5. **Training-set build** — `mask_creator.ipynb` rasterizes GLIMS polygons into masks (reprojecting per
   image CRS); `get_training_set.ipynb` pairs images with masks into `training_data/images` + `masks`.
6. **Train** — `src/segmentation/training/src/train.py`.
7. **Inference** — `src/segmentation/inference/infer.py` (single glacier, GIF + area plot) or
   `src/segmentation/final_areas.py` (every glacier in the landing zone, area CSVs + GIFs).
8. **Analysis** — `areas_true_vs_predicted_final.ipynb` compares predicted areas to GLIMS `db_area`.

## Data: S3 buckets → local paths

Nothing under `data/` is in git. The code reads from a local "landing zone" whose layout is fixed by
`data_label` — a string hardcoded near the top of every script and notebook that selects which dataset
to operate on:

```
src/earth_engine/data/
  ee_landing_zone/<data_label>/
    landsat/<glims_id>/<glims_id>_<date>_<L5|L7|L8>_C02_T1_L2_SR.tif
    landsat/<glims_id>/meta_data/metadata_list_l{5,7,8}   # raw EE getInfo() blobs
    dems/<glims_id>_NASADEM.tif
    logs/log_<name>.log
  processed_metadata/<data_label>/*.csv                    # relationalized metadata
src/glims/data/
  glims_db_20210914/glims_polygons.shp                     # full GLIMS inventory
  training_samples/glims_18k_bb.shp                        # bounding boxes, 18k glaciers
  inference_samples/geog_area_rollup_{50,250}.csv          # which glaciers to infer on
```

Two `data_label`s are live: `localized_time_series_for_training_c02_t1_l2` (training, small crops) and
`full_time_series_c02_t1_l2` (inference, full time series). `full_time_series`,
`localized_time_series_for_segmentation_training`, and `..._large` are earlier, pre-Collection-2
iterations — don't populate them.

### Bucket mapping

Verified against the project AWS account with the AWS CLI (`aws sts get-caller-identity` shows which). Region is split across `us-east-1` and
`us-west-1`, so every call needs the right `--region`.

| Bucket | Region | Holds | Local destination |
|---|---|---|---|
| `full-time-series-images-t1-l2-sr` | us-west-1 | `{glims_id}/*.tif` + `{glims_id}/meta_data/metadata_list_l{5,7,8}`; 1,102 glaciers, **486 GB / 516k objects** | `ee_landing_zone/full_time_series_c02_t1_l2/landsat/` |
| `full-time-series-dems-t1-l2-sr` | us-east-1 | flat `{glims_id}_NASADEM.tif`; 1,122 objects, 89 MB | `ee_landing_zone/full_time_series_c02_t1_l2/dems/` |
| `full-time-series-metadata-t1-l2-sr` | us-east-1 | one prefix per Glue table (`{ee_metadata,file_attributes,image_attributes,processed_logs}_full_time_series[_250]/`) | `processed_metadata/full_time_series_c02_t1_l2/` |
| `raw-training-images-t1-l2-sr` | us-west-1 | `landsat/{glims_id}/*.tif` + `logs/`; 96 GB / 303k objects | `ee_landing_zone/localized_time_series_for_training_c02_t1_l2/landsat/` |
| `training-images-t1-l2-sr` | us-west-1 | **already-preprocessed tensors** — `images/` (7,174 × 128×128×7 float64) + `masks/` (8,902 × 128×128); 7.2 GB | `src/segmentation/training/src/training_data/{images,masks}/` |
| `training-images-metadata-t1-l2-sr` | us-west-1 | `{ee_metadata,file_attributes,image_attributes,processed_logs}/` + `glims_18k/glims_18k.parquet` | `processed_metadata/localized_time_series_for_training_c02_t1_l2/` |
| `*-trient-only` | us-west-1 | single-glacier (Trient, `G007026E45991N`) copies — 1,753 objects / 755 MB. Cheap end-to-end smoke test | either landing zone |
| `athena-output-gv`, `glacier-view-athena-output-us-west-1` | east / west | Athena query results | output only, never an input |
| `mwaismann-gv` | us-west-1 | a `test/` prefix, 2 objects | scratch, ignore |

`training-images-t1-l2-sr` is the important one to know about: despite the name it is **not** raw
imagery, it is the finished training set `train.py` consumes. `raw-training-images-t1-l2-sr` is the raw
Landsat that `mask_creator.ipynb` / `get_training_set.ipynb` turn into it.

Buckets with a `-gv` suffix and 2021–2022 dates (`joined-time-series-gv`, `joined-dems-gv`,
`training-images-gv`, `training-pickles-gv`, `training-metadata-gv`, `segmentation-metadata-gv`,
`image-quality-classifier-gv`) predate the Collection-2 rework and match landing zones and a
`src/image_quality_classifier/` module that no longer exist in the repo. Archive — don't wire them in.

### Athena / Glue

The Glue catalog lives in **us-west-1** (`glacier-view` database) even though the metadata buckets it
points at are in us-east-1 — cross-region by design, both halves of the LLD's claim are true. Tables,
all external over the prefixes above:

- training vintage: `ee_metadata`, `file_attributes`, `image_attributes`, `processed_logs`, `glims_18k`
- inference, 50-per-rollup: `*_full_time_series`
- inference, 250-per-rollup: `*_full_time_series_250`

`docs/low_level_design.md` lists mangled table names (`file_attributesee_metadata_full_time_series_250`) —
the real names are the clean ones above. The committed SQL joins unqualified table names, so run it
against `glacier-view` in us-west-1. `inference_data_query.sql` targets the `_full_time_series` (50)
vintage. **No committed query selects the 250-per-region set**, even though its imagery and Athena
tables exist — an empty placeholder file was removed rather than left to look like a real query.

### What is present, and what is missing

Checked by diffing local inventories against the buckets.

**Present and complete:**

- Ready-to-train data — `training/data/processed_training_data_summer_months/` holds 7,174 images +
  8,902 masks, an exact mirror of `training-images-t1-l2-sr`. `train.py` wants it at
  `training_data/{images,masks}` relative to its cwd, so symlink rather than re-download.
- Raw training data — 18,093 glacier dirs of imagery and DEMs under
  `localized_time_series_for_training_c02_t1_l2/`, plus `masks_staging_2/`. That is 148 glaciers *more*
  than `raw-training-images-t1-l2-sr` holds (see "unbacked" below).
- Inference DEMs — all 1,122, byte-for-byte the same id set as `full-time-series-dems-t1-l2-sr`.
- Inference metadata — the four table CSVs plus `filtered_inference_data.csv`.

**Missing and blocking:**

- `full_time_series_c02_t1_l2/landsat/` is **empty** — every inference image is absent. That tree lived
  on an external SSD (`/Volumes/T7/GlacierView/ee_landing_zone/...`, which
  `time_series_metadata_handler.ipynb` still points at). Nothing needed is missing *upstream*: all 245
  glaciers in `filtered_inference_data.csv`, and all 1,024 in `geog_area_rollup_250.csv`, exist in S3.
- `model` — both inference scripts `torch.load("model")` relative to cwd, and no such file exists.
  Copy or symlink `src/segmentation/unet_summer_model_unfrozen_100` into place.
- `src/segmentation/glacier_areas/` — `final_areas.py` writes `glacier_areas\\{glims_id}_areas.csv`
  into it but never creates it (and the path uses a Windows separator). Every iteration will throw
  into the bare `except`, so the per-glacier CSVs silently never appear while the aggregate ones do.
- **Both inference scripts build the wrong number of input channels** (10 vs the checkpoints' 9), so
  every glacier errors out into the bare `except`. `train.py` is stale the same way. This is a
  three-line fix — see "Model contract".

**Missing, regenerable:** `areas_no_threshold.csv`, `areas_05_thresh.csv`, `areas_binary_05.csv` are
outputs `final_areas.py` creates on startup.

**Recovered (2026-09-19).** The four analysis inputs `areas_true_vs_predicted_final.ipynb`
live in **`data/analysis/`**, committed (the blanket `*.csv` ignore has a negation for that
directory). The notebook resolves them via an `ANALYSIS_DIR` constant derived from its own location,
so it no longer depends on where Jupyter was launched:

| File | Shape | What it is |
|---|---|---|
| `geo_areas.csv` | 18,093 × 32 | the `glims_18k` table itself — `glac_id`, `db_area`, `geog_area`, **`geog_area_rollup`**, `bboxes`, elevations, `src_date`, submitters. Rollups: Asia 12,625 / N. America 3,704 / S. America 872 / Europe 602 / Caucausus 255 / **Oceania 35** |
| `training_data_set.csv` | 10,447 × 12 | output of `training_data_query.sql` — one row per candidate training image with quality metadata + `rank_score`. This is the paper's stale "10,443". The empty-mask variance filter cuts it to the 7,174 actually trained on |
| `training_areas.csv` | 3 × 260 | transposed lookup: ~259 glacier IDs → areas |
| `areas_binary_05_filtered_smooth_25.csv` | 541 monthly dates × **245** glaciers | smoothed `final_areas.py` output, wide format (`pd.date_range('1979-01-01','2024-01-01',freq='MS')` = 541 months) |

⚠️ **The areas CSV covers 245 glaciers; the paper claims 1,083.** So this is the 50-per-region
vintage, not the run behind the paper's headline numbers — that output is still unaccounted for.
Note `geog_area_rollup` here includes **Oceania**, which the SQL treats as southern-hemisphere but
which never appears among the paper's five regions; those 35 glaciers drop out somewhere.

**Present locally but unbacked — in S3 nowhere:**

- ~~Model checkpoints~~ — **backed up 2026-09-19** to
  `s3://segmentation-model-gv/checkpoints/` (us-east-1): both PyTorch checkpoints under
  `pytorch/`, the 18 legacy Keras files under `keras-legacy/`, and a `MANIFEST.md` recording
  channels, hashes and the training recipe. All 20 objects byte-verified. ⚠️ **Bucket versioning is
  still disabled** — an overwrite is unrecoverable, so enable it before re-uploading.
- 148 training glaciers that are local-only (`raw-training-images-t1-l2-sr` has 17,945 of the 18,093).
- All 18,093 training-landing-zone DEMs — there is no training DEM bucket.
- The derived metadata CSVs: `denormalized_metadata.csv`, `filtered_training_data.csv`,
  `filtered_training_data_summer_months.csv`, `filtered_inference_data.csv`,
  `net_new_glims_ids_for_segmentation.csv`. The buckets hold only the four crawled table CSVs.
- `src/glims/data/glims_db_20210914/` — the source GLIMS inventory. Only the derived
  `glims_18k.parquet` is in S3.

### Refilling the inference imagery

The whole bucket is 486 GB, but `filtered_inference_data.csv` narrows inference to **245 glaciers /
14,952 files = 23.7 GB**. `final_areas.py` calls `read.get_rasters(glacier_dir)` with no
`filtered_file_path`, so it reads whatever is on disk — downloading only the filtered files *is* how
the filter gets applied at inference time. Pull that subset, not the bucket.

```bash
# smoke test: one glacier, ~700 MB
aws s3 cp s3://full-time-series-images-t1-l2-sr/G007026E45991N/ \
  src/earth_engine/data/ee_landing_zone/full_time_series_c02_t1_l2/landsat/G007026E45991N/ \
  --recursive --region us-west-1 --exclude "*/meta_data/*" --exclude "*.DS_Store"
```

`docs/low_level_design.md` notes `sync` was too slow for the original upload; `cp --recursive` is what was
used. `.DS_Store` files were uploaded alongside the tifs (some are 660 KB) — exclude them, and exclude
`meta_data/` unless re-running the metadata handler, since `read.get_rasters` only globs `.tif`.

### Path gotchas

- `infer.py` and `final_areas.py` both hardcode `data_label = "full_time_series"` (no `_c02_t1_l2`
  suffix), which points at an empty legacy directory. The notebooks and all the populated data use
  `full_time_series_c02_t1_l2`. Fix the constant or symlink before running.
- A stray `full_time_series_c02_t1_l2/G007026E45991N/` sits beside `landsat/` rather than inside it —
  a Trient-only download that landed one level too high.
- **No model checkpoint exists in S3.** `segmentation-model-gv` holds only `training_data.pickle`,
  `test_data.pickle`, and an older `training_data/{images,masks}` copy — no weights. The 340 MB
  `src/segmentation/unet_summer_model_unfrozen_100` is untracked and local-only, so it is currently
  unbacked; back it up before touching it.

## Commands

```bash
uv sync                                  # builds .venv from pyproject.toml + uv.lock

cd src/segmentation/training/src && uv run python train.py --epochs 10 --lr 0.00001 --decay 0.00001 --batch 2
uv run python src/segmentation/inference/infer.py --glimsid G007026E45991N
cd src/segmentation && uv run python final_areas.py   # no args; whole landing zone
```

`train.py` expects `training_data/{images,masks}` relative to the cwd, writes each run to
`experiments/<counter>/` (hyperparameters, loss plot, dice histogram, `model`), and also overwrites
`src/segmentation/inference/model` with the newly trained model. **The `--batch 2` default is not
what the paper used — it reports batch size 32.** See the recipe table under "Model contract".

Linting: `uv run ruff check .` — passes repo-wide, config in `pyproject.toml` with an explicitly
pinned rule set. Notebooks have documented per-file exemptions. `ruff format` is deliberately not
enforced. There is still no test suite and no CI.

## Model contract

The published model is described in the project's IEEE paper (see "The paper" below). The U-Net is
defined **three times** — `training/src/train.py`, `inference/cnn.py`, and inlined inside
`inference/infer.py`.

### The checkpoints take 9 input channels, not 10

Verified by inspecting the serialized tensors in both checkpoints (stem conv is
`Conv2d(9,64,7,7)` = 112,896 bytes; `c5` is `conv_block(41,16)` = 23,616 bytes — the 10-channel
variants would be 125,440 and 24,192 and appear in neither file):

```
0: NDWI   1: NDSI   2: blue   3: green   4: red   5: nir   6: swir   7: thermal   8: DEM
```

`swir_2` was **deliberately dropped for quality reasons** — the paper's channel figure says so
explicitly ("Remove SWIR2 because of quality"), and its caption reads "The 9 channels used in the
CNN." Several source files were never updated to match, so **most of the code is stale against the
checkpoints**:

| File | Channels it builds | Against a 9-ch checkpoint |
|---|---|---|
| `get_training_set.ipynb` | 6 bands + DEM = 7 → +NDSI +NDWI = **9** | ✅ correct |
| training data on disk | 7-channel tifs (verified: `SamplesPerPixel: 7`) → **9** | ✅ correct |
| `final_areas.py:101` | 6 bands + DEM = 7 → +NDSI +NDWI = 9, **then a duplicated NDWI = 10** | ❌ shape error |
| `inference/infer.py:39` | 7 bands *including* `swir_2` + DEM = 8 → **10** | ❌ shape error |
| `train.py` | hardcodes `nn.Conv2d(10, …)`; normalize indexes `image[0]`…`image[7]` (8 ch) | ❌ stale |

So `get_training_set.ipynb` and the on-disk training data are the **correct** pair. The fixes:

- **`final_areas.py`** — delete the duplicated-NDWI line. Its own comment already says to
  (`# REMOVE THIS LINE - only for testing`); removing it takes 10 → 9 and makes the script correct.
- **`infer.py`** — drop `'swir_2'` from `common_bands`, and change the inlined `Conv2d(10, …)` to 9.
- **`train.py`** — change `Conv2d(10, …)` to 9, and fix the normalize block to 7 channels not 8.

Note the hardcoded `10` in the class definitions does **not** break loading: `torch.load` of a
pickled module restores the real layer shapes and never re-runs `__init__`. It only matters if you
construct a fresh `UNet()` to train.

### Checkpoints are pickled modules referencing `__main__`

`torch.save(model)` saved the whole live object from a script, so the pickle references
`__main__.UNet` and `__main__.conv_block` (verified by reading the pickle's string table). To load a
checkpoint, those two names must exist in the **`__main__`** namespace — which is why `infer.py`
inlines a copy of the class, and why `final_areas.py`'s `from inference.cnn import UNet, conv_block`
works only because it is itself `__main__` when run as a script. Loading from a notebook or library
needs the same names injected first. Consequences:

- Edit `UNet`/`conv_block` and old checkpoints may stop loading. Change all three copies together.
- `torch.load("model")` is **cwd-relative** in both scripts and no file named `model` exists — copy
  or symlink a checkpoint first.
- Converting to a `state_dict` is what makes these portable; do that before sharing them anywhere.

### Other invariants

- ResNet-50 stem replaced with a 9-channel `Conv2d`; ImageNet weights loaded for the rest; encoder
  output is 2048 channels at 4×4. Last two ResNet layers removed.
- Output: 2-class logits (glacier / background, complementary). Probability is
  `softmax(...)[:, 1]`, thresholded at 0.5.
- Area = `mask.sum() * 0.0009` km² (30 m × 30 m pixels), computed **after** resizing the prediction
  back to the image's original dimensions.
- NDSI/NDWI are computed from the raw band stack at fixed indices (`green=ch1`, `nir=ch3`,
  `swir=ch4`) then *prepended*, NDSI first and NDWI second — so NDWI lands at index 0.

### Two checkpoints exist

| File | Date | Size | Notes |
|---|---|---|---|
| `src/segmentation/unet_summer_model_unfrozen_100` | 2024-10-02 | 343,522,287 B | the production checkpoint; 9-channel |
| `src/segmentation/saved_models/model` | 2023-12-27 | 343,487,293 B | earlier, distinct SHA-256; also 9-channel |

`saved_models/` also holds 18 legacy Keras `.h5` files (2022–2023) whose names record the band
experiments (`bl_gr_re_ni_sw_th_de_v{0,1,2}.h5`, `re_ni_sw_de_v1.h5`, …). **No model weights exist in
S3** — `segmentation-model-gv` holds only data pickles.

### Published training recipe (from the paper — differs from the script defaults)

| | Paper | `train.py` default |
|---|---|---|
| learning rate | 0.00001 | 0.00001 ✅ |
| **batch size** | **32** | **2** ❌ |
| encoder | unfrozen | `UNET(freeze_encoder=False)` ✅ |
| loss | cross-entropy | `LOSS = 'ce'` ✅ |
| threshold | 0.5 | 0.5 ✅ |
| optimizer | Adam | Adam ✅ |
| splits | 6,456 train / 359 eval / 359 test of 7,174 | 90/10 only, no eval set |
| model selection | min cross-entropy on the eval set | last epoch |

Augmentation: vertical flip, horizontal flip, and 3×3 Gaussian blur (σ = 0.8), each applied
independently with p = 0.5 — growing the training set ~75%, to 11,298 images. Reported test
**Dice = 0.92**.

Ablations the paper records: 3,000 images without transfer learning 0.83 → with transfer learning
0.86 → adding NDSI/NDWI 0.88 → full dataset with cross-entropy **0.92** (Dice loss gave 0.90).
Frozen encoder 0.91 vs unfrozen 0.92. ResNet-50 beat ResNet-18 by 1–2%. Threshold sweep:
0.1 → 0.16, 0.4 → 0.923, **0.5 → 0.926**, 0.8 → 0.87.


## Raster conventions

`helpers/read.py` + `helpers/preprocess.py` are the shared raster path used by every notebook and script:

- `read.get_rasters` returns a `{file_name: HWC array}` dict, clamping `-inf` and negatives to 0.
- Band indices differ per satellite — `helpers/landsat_bands.py` maps names to indices for `l5`/`l7`/`l8`,
  and `preprocess.get_common_bands` uses the **third underscore-delimited token of the filename**
  (`{glims_id}_{date}_{L5|L7|L8}_C02_T1_L2_SR.tif`) to pick the right map. Filenames are load-bearing;
  `infer.py` also parses the date from token 1.
- `preprocess.normalize_rasters` normalizes per band using the *second* min/max (the extremes are
  treated as sensor artifacts and replaced) — not a plain min-max.
- Everything is resized to 128×128 before the model and resized back afterwards for area.

Two steps the paper documents that are easy to miss in the code:

- **Training images were temporally averaged.** All Landsat scenes within a ±100-day window of the
  GLIMS `src_date` (the outline's as-of date) were pulled and combined by arithmetic mean, to
  suppress shadows, debris and defects. So one training image is a composite, not a single scene.
- **Inference applies a Gaussian filter along the time axis**, not in space:
  `gaussian(X, sigma=[20,0,0,0], mode='reflect')` — σ = 20 *time points*, smoothing each pixel
  across the glacier's time series to reduce noise.

⚠️ The two inference scripts disagree on where smoothing happens, and the paper sides with
`infer.py`: the paper smooths first and computes NDSI/NDWI on the smoothed stack (as `infer.py`
does), while `final_areas.py` computes the indices first and then smooths everything including them.

⚠️ Reprojection direction also differs: the paper says the *rasters* were reprojected to WGS 84 to
match the polygons, while `mask_creator.ipynb` reprojects the *polygons* into each raster's UTM CRS.
Same alignment goal, opposite direction — likely an evolution, but don't assume either doc matches
what produced a given artifact.

## Repo conventions and gotchas

- Notebooks hardcode `~/Desktop/projects/GlacierView` as the project root and `sys.path.insert` the
  helpers directory. Some also point at an external SSD (`/Volumes/T7/...`) for the image landing zone.
  Scripts under `src/segmentation/` use `Path(__file__).parent...` instead.
- `src/earth_engine/` and all `data/` directories are untracked/gitignored — image, DEM, CSV, PNG, and
  GIF outputs never get committed. Expect the landing zone to be absent on a fresh clone.
- `helpers/model.py` and `inference/model.py` are dead TensorFlow/Keras code from an earlier iteration.
  The live model is PyTorch (`inference/cnn.py`). Don't extend the Keras files.
- `final_areas.py` wraps each glacier in a bare `except:` that prints `Error {glims_id}` — failures are
  silent by design during long batch runs; add logging rather than assuming a clean run succeeded.
- Plots use the shared style `src/styles/ieee.mplstyle`.
