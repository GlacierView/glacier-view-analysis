# Data model

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

Bucket inventory and the Athena catalog: `.agents/references/aws.md`.

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
