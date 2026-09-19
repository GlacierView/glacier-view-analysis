# Run inference

## When to use

To segment glaciers and produce surface-area time series, either for one glacier or in batch.

## One glacier

```bash
uv run python src/segmentation/inference/infer.py --glimsid G007026E45991N
```

Writes a GIF and an area plot to `src/segmentation/gifs/`. Needs a checkpoint: both entry points
`torch.load` a path relative to the working directory, and no file named `model` exists, so copy or
symlink one first.

## Every glacier in the landing zone

```bash
cd src/segmentation && uv run python final_areas.py
```

No arguments. Writes three aggregate CSVs (raw, thresholded, binarized), per-glacier CSVs under
`glacier_areas/`, and GIFs. The aggregates are rewritten after each glacier, so a run that dies
partway still leaves usable output.

**Check the failure tally it prints.** Each glacier is wrapped in a `try`/`except` that logs a
traceback and continues, so a run can complete having failed on most of them.

## Getting the imagery first

The landing zone's `landsat/` directory is empty locally. Start with one glacier (~700 MB):

```bash
aws s3 cp s3://full-time-series-images-t1-l2-sr/G007026E45991N/ \
  src/earth_engine/data/ee_landing_zone/full_time_series_c02_t1_l2/landsat/G007026E45991N/ \
  --recursive --region us-west-1 --exclude "*/meta_data/*" --exclude "*.DS_Store"
```

The whole bucket is 486 GB; the filtered inference set is 245 glaciers / 14,952 files / 23.7 GB.
Pull that subset, not the bucket. See `.agents/rules/data.md`.

## Verifying it worked

Trient (`G007026E45991N`) is the documented regression case: 464 images, 1984-04-18 to 2023-10-07,
decline concentrated in 1985–1988 and 2002–2015, roughly stable over the last decade. Compare
against the previous runs' GIFs already in `src/segmentation/gifs/`.

## References

- `.agents/context/model.md` — what the model expects
- `.agents/references/aws.md` — bucket inventory
- `docs/ONBOARDING.md` — the smoke-test walkthrough and Trient caveats
