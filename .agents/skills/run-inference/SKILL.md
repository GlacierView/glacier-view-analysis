# Run inference

## When to use

To segment glaciers and produce surface-area time series, for one glacier or
the whole landing zone.

## One glacier

```bash
uv run glacierview infer --glims-id G007026E45991N \
  --checkpoint src/segmentation/unet_summer_model_unfrozen_100
```

Writes a GIF and an area plot under the output directory. Use
`glacierview config` first if a path looks wrong — it prints everything
resolved, and flags what is missing.

## Every glacier in the landing zone

```bash
uv run glacierview areas --checkpoint <path>
uv run glacierview areas --checkpoint <path> --glims-id G007026E45991N --no-gif
```

Writes three aggregate CSVs (raw, thresholded, binary), per-glacier CSVs under
`glacier_areas/`, and GIFs. Aggregates are rewritten after each glacier, so a
run that dies partway still leaves usable output.

**Read the summary it prints.** Failures are logged with a traceback and
counted, not raised — with hundreds of glaciers one bad scene should not cost
the run. The exit code is non-zero if anything failed.

## Getting the imagery first

The landing zone's `landsat/` directory is empty locally. Start with one
glacier (~700 MB):

```bash
aws s3 cp s3://full-time-series-images-t1-l2-sr/G007026E45991N/ \
  "$(uv run glacierview config | awk '/landsat_dir/{print $2}')/G007026E45991N/" \
  --recursive --region us-west-1 --exclude "*/meta_data/*" --exclude "*.DS_Store"
```

The whole bucket is 486 GB; the filtered inference set is 245 glaciers /
14,952 files / 23.7 GB. Pull that subset. See `.agents/rules/data.md`.

## Verifying it worked

Trient (`G007026E45991N`) is the documented regression case: 464 images,
1984-04-18 to 2023-10-07, decline concentrated in 1985–1988 and 2002–2015,
roughly stable over the last decade.

## References

- `.agents/context/model.md` — what the model expects
- `.agents/references/aws.md` — bucket inventory
