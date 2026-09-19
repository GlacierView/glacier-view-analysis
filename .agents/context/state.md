# Current state of the data

Checked by diffing local inventories against the buckets.

**Present and complete:**

- Ready-to-train data — `training/data/processed_training_data_summer_months/` holds 7,174 images +
  8,902 masks, an exact mirror of `training-images-t1-l2-sr`. `glacierview train` wants it at
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
- `src/segmentation/glacier_areas/` — `glacierview areas` writes `glacier_areas\\{glims_id}_areas.csv`
  into it but never creates it (and the path uses a Windows separator). Every iteration will throw
  into the bare `except`, so the per-glacier CSVs silently never appear while the aggregate ones do.
- ~~Input-channel mismatch~~ — fixed. All entry points now derive their channel count from
  `IN_CHANNELS` in `glacierview.models.unet` and assert it before the first convolution.

**Missing, regenerable:** `areas_no_threshold.csv`, `areas_05_thresh.csv`, `areas_binary_05.csv` are
outputs `glacierview areas` creates on startup.

**Recovered (2026-09-19).** The four analysis inputs `areas_true_vs_predicted_final.ipynb`
live in **`data/analysis/`**, committed (the blanket `*.csv` ignore has a negation for that
directory). The notebook resolves them via an `ANALYSIS_DIR` constant derived from its own location,
so it no longer depends on where Jupyter was launched:

| File | Shape | What it is |
|---|---|---|
| `geo_areas.csv` | 18,093 × 32 | the `glims_18k` table itself — `glac_id`, `db_area`, `geog_area`, **`geog_area_rollup`**, `bboxes`, elevations, `src_date`, submitters. Rollups: Asia 12,625 / N. America 3,704 / S. America 872 / Europe 602 / Caucausus 255 / **Oceania 35** |
| `training_data_set.csv` | 10,447 × 12 | output of `training_data_query.sql` — one row per candidate training image with quality metadata + `rank_score`. This is the paper's stale "10,443". The empty-mask variance filter cuts it to the 7,174 actually trained on |
| `training_areas.csv` | 3 × 260 | transposed lookup: ~259 glacier IDs → areas |
| `areas_binary_05_filtered_smooth_25.csv` | 541 monthly dates × **245** glaciers | smoothed `glacierview areas` output, wide format (`pd.date_range('1979-01-01','2024-01-01',freq='MS')` = 541 months) |

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
