# Data rules

- Never commit imagery, DEMs, model checkpoints or generated output. The `data/` directories under
  `src/` are gitignored, and a fresh clone has no landing zone. The exceptions, deliberately
  tracked, are `data/analysis/` and `figures/`.
- Scope S3 pulls. The inference bucket is 486 GB, but the filter narrows it to 245 glaciers /
  23.7 GB. `final_areas.py` reads whatever is on disk, so downloading only the filtered files *is*
  how the quality filter gets applied. Exclude `.DS_Store` (some are 660 KB) and `meta_data/`.
- Every `aws s3` call needs the right `--region`; buckets are split across `us-east-1` and
  `us-west-1`. See `.agents/references/aws.md`.
- A stray `full_time_series_c02_t1_l2/G007026E45991N/` sits beside `landsat/` rather than inside it
  — an old Trient download one level too high. Don't mistake it for the landing zone.
- Notebooks hardcode `~/Desktop/projects/GlacierView` as the project root and `sys.path.insert` the
  helpers directory; some point at an external SSD (`/Volumes/T7/...`). Scripts under
  `src/segmentation/` use `Path(__file__)` instead and work from any directory.
