# Pipeline

Data flows in one direction; each stage writes files that the next stage reads.

1. **GLIMS selection** — `src/glims/src/` notebooks read the GLIMS polygon shapefile, roll `geog_area`
   up into 5 continental buckets (`geog_area_rollup`), and emit bounding boxes + sample CSVs.
2. **Earth Engine pull** — `glacierview.earthengine.export` (`EePull` class) downloads
   one GeoTIFF per image via `geemap.ee_export_image` at 30 m scale, plus a NASADEM per glacier.
   Server-side filters: `CLOUD_COVER <= 10`, `IMAGE_QUALITY == 9` (`IMAGE_QUALITY_OLI` for L8).
   Each image's `getInfo()` JSON is appended to `meta_data/metadata_list_l{5,7,8}` alongside the tifs,
   and a log line per image goes to the run log.
3. **Metadata relationalization** — `src/earth_engine/src/metadata/time_series_metadata_handler.ipynb`
   parses the logs, dedupes the EE metadata blobs (EE sometimes returns >1 record per image), opens each
   tif with rasterio for image/file attributes, and writes CSVs → S3 → AWS Glue → Athena
   (database `glacier-view` in us-west-1 — see "Athena / Glue" below).
4. **SQL filtering** — `sql/` queries run against Athena to pick which GLIMS IDs and which individual
   files are eligible. `identify_inference_*` picks glaciers (N per `geog_area_rollup`);
   `inference_data_query.sql` / `training_data_query.sql` pick files. Criteria: cloud cover < 10%,
   summer months only (May–Oct northern hemisphere, Nov–Apr southern), > 50,000 pixels,
   0 no-data pixels, < 10% zero pixels.
5. **Training-set build** — `mask_creator.ipynb` rasterizes GLIMS polygons into masks (reprojecting per
   image CRS); `get_training_set.ipynb` pairs images with masks into `training_data/images` + `masks`.
6. **Train** — `glacierview train`.
7. **Inference** — `src/segmentation/inference/infer.py` (single glacier, GIF + area plot) or
   `glacierview areas` (every glacier in the landing zone, area CSVs + GIFs).
8. **Analysis** — `areas_true_vs_predicted_final.ipynb` compares predicted areas to GLIMS `db_area`.
