# AWS inventory

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
imagery, it is the finished training set `glacierview train` consumes. `raw-training-images-t1-l2-sr` is the raw
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

## Athena query history

`athena:ListQueryExecutions` and `athena:ListNamedQueries` are **denied** to the
team IAM role, so past query text cannot be read via the API. Query *results*
are readable, in the two output buckets under `Unsaved/<yyyy>/<mm>/<dd>/`, as a
`<uuid>.csv` plus a `<uuid>.csv.metadata` that holds column types only — not
the SQL.

That is still enough to identify a run. The 250-per-region glacier selection
was recovered this way: `Unsaved/2024/09/11/21704e69-…csv` is byte-identical to
`geog_area_rollup_250.csv`, which pinned every parameter of the query now
reconstructed in `sql/identify_inference_glims_ids_geog_area_rollup_250.sql`.
