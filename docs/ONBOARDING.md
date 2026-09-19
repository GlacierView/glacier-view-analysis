# GlacierView — Onboarding

**Audience:** someone new to this project who has never worked with satellite imagery before.
No remote-sensing background assumed. Read Parts 1–2 before touching any code.

**Status of this document:** written 2026-09-19 by reading every source file, the design doc, and the
live AWS account. Where a claim was checked against something runnable, it
says so. Where it's an inference, it says that too. `low_level_design.md` (the "LLD") is the original
author's design doc and is still the best source for *why* decisions were made — but several of its
concrete details have drifted from the code, and those are flagged throughout.

---

## Table of contents

1. [What this project does](#part-1--what-this-project-does)
2. [Vocabulary you need](#part-2--vocabulary-you-need)
3. [The pipeline, end to end](#part-3--the-pipeline-end-to-end)
4. [Code tour: what every file is](#part-4--code-tour-what-every-file-is)
5. [The AWS account](#part-5--the-aws-account)
6. [The data model](#part-6--the-data-model)
7. [The model contract](#part-7--the-model-contract)
8. [Data state, and what is still open](#part-8--data-state-and-what-is-still-open)
9. [Getting set up](#part-9--getting-set-up)
10. [Landmines, ranked](#part-10--landmines-ranked)
11. [Glossary](#part-11--glossary)

---

## Part 1 — What this project does

### The question

Glaciers are shrinking. How fast, for which glaciers, and can we measure it automatically from
free satellite imagery instead of sending people to survey ice?

### The approach in four sentences

1. Take a public inventory of the world's glaciers — about 18,000 of them, each with a hand-drawn
   outline and an ID.
2. For each glacier, download every usable Landsat satellite photo of it from 1984 to today.
3. Run each photo through a neural network that labels every pixel "ice" or "not ice."
4. Count ice pixels per photo, multiply by the area of a pixel, and you have a surface-area
   measurement per glacier per date — a time series of glacier retreat.

The output artifacts are: per-glacier area time-series CSVs, animated GIFs showing the ice boundary
moving over 40 years, and an analysis comparing the model's areas to the inventory's published areas.

### The paper, and the headline result

This work is written up as an IEEE journal submission: **"A Large Scale Analysis of Mountain Glacier
Shrinkage Using Convolutional Neural Networks"**, by Matthew Waismann (Amazon), Somansh Budhwar, and
Armin Schwartzman (Halıcıoğlu Data Science Institute, UC San Diego — `armins@ucsd.edu`). The LaTeX
source is not in this repo; ask a team member for it. Read the abstract and the Results section
before the code — it tells you what the code was *for*.

What it claims:

- Trained on **7,174** glacier outlines and their images, one time point each.
- **Dice score 0.92** on the held-out test set.
- Applied to **505,835 images across 1,083 glaciers**, spanning **1984–2023**.
- Overall glacier surface-area decline of **0.2% ± 0.01% per year**, negative and statistically
  significant in all five regions.

Per-region relative area change per year, by log-linear regression:

| Region | Change/yr | Std. error |
|---|---|---|
| Europe | −0.00256 | 0.00019 |
| South America | −0.00256 | 0.00029 |
| Caucasus | −0.00167 | 0.00025 |
| Asia | −0.00163 | 0.00015 |
| North America | −0.00155 | 0.00017 |
| **Total** | **−0.00199** | **0.00010** |

Cumulative declines 1985–2022 from the bootstrap analysis: Europe 16.2%, Caucasus 9.1%, Asia 7.1%,
South America 2.6%, North America 2.3%. About two-thirds of individual glaciers shrank; some grew.

**Numbers worth memorizing**, because they recur everywhere and let you tell datasets apart at a
glance: **328,115** glaciers in GLIMS → **18,093** after selection → **10,447** candidate training
images → **7,174** actually trained on → **1,083** glaciers for inference → **505,835** images.

The paper is a live draft with `\editcom{}` review notes in it, and two of its own numbers are
stale: the conclusion says "a dataset of 10,443 glaciers" (that's the 10,447-row
`training_data_set.csv`, and they're images not glaciers), and its body text says 10 input channels
where its own channel figure says 9. The figure is right — see Part 7, this matters a lot.

### Why this is harder than it sounds

- **Clouds.** Most satellite photos of mountains are partly cloudy. Clouds are white. So is ice.
- **Snow.** Seasonal snow is also white and also not glacier. The project only uses summer images
  (when seasonal snow has melted) to reduce this.
- **The satellites changed.** Three different Landsat satellites cover 1984–present, and they have
  *different sensors with bands in different orders*. Normalizing across them is a real chunk of
  this codebase.
- **Map projections.** The Earth is round, images are flat. The same glacier photographed on two
  different dates can come back in two different coordinate systems, so the hand-drawn outline
  doesn't line up with the pixels without reprojection work.
- **Scale.** ~500,000 satellite images, ~490 GB. You cannot hold this in memory or on a laptop.

### Project shape, honestly

This is **research code by a small team**, not a production system. There are no tests, no CI, no
linter, no packaging. Notebooks hardcode one person's home directory. Several scripts have
copy-pasted duplicate model definitions. There is a bare `except:` that swallows every error in the
main inference loop. Understanding *that* is part of onboarding: when something doesn't work here,
the default assumption should be "the code drifted from the data," not "I'm using it wrong."

---

## Part 2 — Vocabulary you need

Skim this, then come back when a term bites you.

### Glacier inventories

**GLIMS** (Global Land Ice Measurements from Space) is a public database of glacier outlines. Each
glacier has:
- a **GLIMS ID** like `G007026E45991N` — that's `G` + longitude `007.026°E` + latitude `45.991°N`,
  encoded into the ID. So the ID tells you where the glacier is. `G007026E45991N` is the Trient
  glacier in the Swiss Alps and is this project's default test glacier.
- a **polygon** — the hand-drawn outline of the ice, in latitude/longitude.
- **`db_area`** — the surface area in km² recorded in the inventory, used as ground truth to compare
  the model against.
- **`geog_area`** — who submitted the outline and for what region ("Swiss Alps", "China",
  "Cordillera Blanca, Peru"). Free text, messy. The project rolls these up into five buckets called
  **`geog_area_rollup`**: Asia, Europe, North America, South America, Caucausus (sic — the
  misspelling is in the data and the code, so keep it).

**RGI** (Randolph Glacier Inventory) is a different inventory that got merged into GLIMS. It appears
as a `geog_area` value and is deliberately excluded here, because its outlines are lower quality for
this purpose. See `figures/inventory-examples/` for screenshots of why.

### Satellites and imagery

**Landsat** is a series of US government Earth-observation satellites. Free data, and critically a
*continuous record since 1972*, which is what makes 40-year time series possible. This project uses:

| Satellite | Active | Sensor note |
|---|---|---|
| Landsat 5 | 1984–2013 | the backbone of the early record |
| Landsat 7 | 1999–present | has a hardware fault since 2003 that leaves diagonal gaps in images |
| Landsat 8 | 2013–present | newer sensor, **different band order** — this matters a lot |

A Landsat image is **30 metres per pixel**. One pixel = 30 m × 30 m = 900 m² = **0.0009 km²**. That
constant is how pixel counts become areas, and you'll see `* 0.0009` in the code.

**`LANDSAT/LT05/C02/T1_L2`** — this is an asset ID you'll see everywhere. Decoded:
- `LT05` = Landsat 5 (`LE07` = 7, `LC08` = 8)
- `C02` = **Collection 2**, the current reprocessing of the whole archive. Collection 1 is obsolete.
  A lot of legacy data in this project is pre-Collection-2 and should not be mixed with new data.
- `T1` = **Tier 1**, the highest-quality geometrically-corrected subset.
- `L2` = **Level 2**, meaning **surface reflectance** — atmospheric effects already removed.
  (Level 1 is raw "top of atmosphere"; you want L2 for comparing across dates.)

You'll also see `_SR` in filenames, for the same reason (Surface Reflectance).

### Bands

A satellite doesn't take one photo, it takes several simultaneously at different wavelengths. Each
is a **band**. A Landsat GeoTIFF here is a stack of bands, like a very tall RGB image.

The bands this project cares about:

| Band | What it sees | Why it's here |
|---|---|---|
| blue, green, red | visible light | human-viewable image; ice is bright in all three |
| **nir** (near infrared) | just past red | vegetation is very bright; ice is dark |
| **swir** (shortwave infrared) | further out | **ice and snow are very dark here, clouds are bright** — the single most useful band for telling ice from cloud |
| swir_2 | a second SWIR window | more of the same |
| thermal | heat | ice is cold |
| `atmos_opacity`, `cloud_qa`, `qa_aerosol`, … | sensor quality flags | present in the files; **this project ignores them** |

**Critically: the band order differs per satellite.** On Landsat 5 and 7, band index 0 is blue. On
Landsat 8, index 0 is `coastal_aerosol` and blue is index 1 — everything shifts by one.
`glacierview.bands` is the lookup table that fixes this, and
`preprocess.get_common_bands()` uses it to produce a consistent band order regardless of satellite.
This is why **filenames are load-bearing** — see Part 6.

### Spectral indices (NDSI, NDWI)

An **index** is a cheap arithmetic combination of two bands that makes one thing pop out. The
standard form is a *normalized difference*: `(A - B) / (A + B)`, which always lands in [-1, 1].

- **NDSI** — Normalized Difference **Snow** Index — `(green - swir) / (green + swir)`.
  Snow/ice is bright in green and dark in SWIR, so NDSI is **high for ice**. This is the classic
  glacier-detection index and doing it by threshold alone was the pre-deep-learning approach.
- **NDWI** — Normalized Difference **Water** Index — `(green - nir) / (green + nir)`.
  High for water. Useful because glacial meltwater lakes are a common false positive.

The project computes both and feeds them to the network as extra input channels, so the network
starts with the physics already half-done for it.

### DEMs

A **DEM** (Digital Elevation Model) is a raster where each pixel's value is the *ground elevation*
in metres, not a colour. **NASADEM** is the specific free global DEM used here. It's fed to the model
as another channel, on the reasoning that glaciers occur at predictable altitudes and slopes.

Note a DEM is **static** — one per glacier, reused for every date in that glacier's time series.

### Files, projections, and why the same glacier moves

**GeoTIFF** — a TIFF image with geographic metadata baked in: which coordinate system it's in, and
which real-world coordinates the corners map to. Read in Python with **rasterio**.

**CRS** (Coordinate Reference System) — the recipe mapping pixel positions to Earth positions,
identified by an **EPSG code**. Two you'll meet:
- `EPSG:4326` — plain latitude/longitude. GLIMS polygons are in this.
- `EPSG:326xx` — **UTM zones**. UTM slices the planet into 60 north-south strips and gives each its
  own flat projection, so distances are accurate locally. Landsat images come in UTM.

**The UTM gotcha, which cost this project real time:** a glacier sitting on a zone boundary can be
delivered in zone 31 on one date and zone 32 on another. Same ice, different coordinate numbers. So
turning a lat/long GLIMS polygon into a pixel mask requires knowing *that specific image's* CRS.
`mask_creator.ipynb` builds a glacier→CRS lookup from the metadata to handle this, and
`figures/utm zone 31-32 overlap trient landsat 7-5.png` is a picture of the problem.

### Image-quality metadata

Every Landsat image ships with metadata. The four fields that drive all the filtering here:
- **`CLOUD_COVER`** — percent of the scene under cloud, 0–100.
- **`IMAGE_QUALITY`** (or `IMAGE_QUALITY_OLI` on Landsat 8) — the sensor's own 0–9 self-assessment.
  `9` is the best. The project accepts nothing less.
- **no-data pixels** — pixels with no measurement at all (off the edge of the satellite's swath, or
  in Landsat 7's fault stripes). The project requires **zero** of these.
- **zero pixels** — pixels that are literally 0, usually a sign of a bad or clipped read. The
  project allows under 10%.

### Machine learning terms

**Semantic segmentation** — classifying *every pixel* of an image, rather than labelling the image
as a whole. Output is a **mask**: same height and width as the input, one class label per pixel.
Here there are two classes: not-glacier (0) and glacier (1).

**U-Net** — the standard architecture for this. Two halves:
- an **encoder** that progressively shrinks the image while extracting increasingly abstract
  features ("there is a bright cold region here"),
- a **decoder** that expands back up to full resolution to produce the per-pixel mask,
- plus **skip connections** wiring each encoder level straight across to the matching decoder
  level. Those are the crossbars of the "U" and they're what preserves fine boundary detail.

**ResNet-50** — a well-known 50-layer image network. Here it's used as the U-Net's encoder, loaded
with **ImageNet weights** — i.e. pretrained on millions of ordinary photographs. This is **transfer
learning**: general-purpose edge/texture detectors learned from photos of cats transfer usefully to
satellite imagery, so you need far less glacier training data. One wrinkle: ResNet-50 expects 3
channels (RGB) and we have 10, so its very first layer is surgically replaced (see Part 7).

**Freezing** — holding the pretrained encoder's weights fixed while only the decoder learns. The
production checkpoint is named `unet_summer_model_unfrozen_100`, which reads as: trained on summer
images, encoder **un**frozen (so everything was fine-tuned), 100 epochs.

**Dice score** and **Jaccard / IoU** — the two standard "how much do the predicted and true masks
overlap" metrics, both 0–1 where 1 is perfect. Plain accuracy is useless here: if a glacier is 5% of
the image, predicting "no ice anywhere" scores 95% accuracy.

**Threshold** — the network outputs a *probability* per pixel. `PROB_THRESH = 0.5` turns
probabilities into a hard yes/no mask.

---

## Part 3 — The pipeline, end to end

Eight stages. Each writes files that the next reads — there is no orchestrator, no DAG, no Makefile.
You run stages by hand, in order. Stages 1–5 have been run already; their outputs are what's on disk.

```
                    ┌─────────────────────────────────────────┐
  GLIMS shapefile → │ 1. Select glaciers, compute bounding box │ → glims_18k_bb.shp
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 2. Download from Google Earth Engine    │ → .tif per image + logs + EE JSON
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 3. Turn metadata into tables → S3       │ → 4 CSVs → AWS Glue → Athena
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 4. SQL: pick good glaciers & good files │ → filtered_*.csv
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 5. Build training set (images + masks)  │ → training_data/{images,masks}
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 6. Train the U-Net                      │ → a checkpoint file
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 7. Inference over time series           │ → area CSVs + GIFs
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │ 8. Analysis: predicted vs. GLIMS areas  │ → plots
                    └─────────────────────────────────────────┘
```

### Stage 1 — Select glaciers

**Where:** `notebooks/pipeline/01_glims_select_glaciers.ipynb`, then `glims_to_parquet.ipynb`

Reads the full GLIMS shapefile (`src/glims/data/glims_db_20210914/glims_polygons.shp`) and filters
it down to a workable set. The actual filters, read off the notebook:

- outline drawn **after 2005-01-01** (older outlines are less reliable)
- **exclude** these `geog_area` values: `Svalbard, Jan Mayen, and Bouvet`,
  `Various (GlobGlacier)`, the `Randolph Glacier Inventory; Umbrella RC…` bucket,
  `Antarctic Peninsula`, `Various (NSIDC)`
- **`db_area` between 1 and 50 km²** — nothing too small to resolve at 30 m, nothing so large it
  won't fit a manageable image
- **de-duplicate `glac_id`** — GLIMS sometimes has several outlines for one glacier; keep the first

What survives is ~18,000 glaciers, hence "18k" in filenames.

Then for each glacier it computes a **bounding box**: the polygon's envelope scaled up by 1.1× in
both axes, so there's a 10% margin of non-glacier context around the ice. That box is what gets
handed to Earth Engine as the download region. Output: `glims_18k_bb.shp` (with a `bboxes` column)
and `glims_18k.parquet` (geometry dropped, because polygons don't survive the Parquet conversion —
the parquet exists purely so Athena can query glacier attributes).

### Stage 2 — Download imagery from Google Earth Engine

**Where:** `glacierview.earthengine.export` + `notebooks/pipeline/03..05_ee_pull_*.ipynb`

**Google Earth Engine (GEE)** is Google's hosted planetary imagery archive. You don't download the
archive; you send it a query and it returns just your clipped region. Requires a Google account with
EE access and `ee.Authenticate()` once.

The class `EePull` has one method per satellite (`export_landsat_five_images`, `…seven…`, `…eight…`)
plus `export_nasa_dems`. Each builds an **ImageCollection** and filters it *server-side* — which
matters, because filtering server-side means you never pay to download images you'd throw away:

```python
ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
  .filterBounds(region)              # the glacier's bounding box
  .filterDate(start_date, end_date)
  .filter(ee.Filter.lte('CLOUD_COVER', 10))
  .filter(ee.Filter.eq('IMAGE_QUALITY', 9))      # IMAGE_QUALITY_OLI for Landsat 8
```

Per the LLD these filters cut the collection by about **63%**. The LLD also notes, self-critically,
that the summer-month filter *could* have been pushed server-side too and wasn't ("… yeah probably,
next time") — so summer filtering happens later, in SQL, after paying to download winter images.

Three things come out of each image, all at once:
1. the **GeoTIFF** itself, at `scale=30` (metres/pixel), all bands in one file
2. the raw **Earth Engine metadata** — `image.getInfo()` returns a JSON blob, and these get
   accumulated into `meta_data/metadata_list_l5` / `_l7` / `_l8` next to the tifs
3. a **log line** per image, so you can `tail -f` a 3-day download and see it's still alive

### Stage 3 — Turn metadata into queryable tables

**Where:** `notebooks/pipeline/07_metadata_time_series.ipynb`
(and `training_metadata_processor.ipynb` for the training vintage)

Now you have half a million tifs and a pile of JSON. This stage makes it queryable. It produces
**four CSVs**, each one row per image file, joined on `file_name`:

| CSV | Source | Contents |
|---|---|---|
| `processed_logs` | the download logs | glims_id, CRS, cloud cover, scene time, image quality |
| `ee_metadata` | the JSON blobs | everything Earth Engine knows about the scene |
| `file_attributes` | the filesystem | file name, date, satellite, size in bytes |
| `image_attributes` | opening each tif with rasterio | height, width, band count, bounds, EPSG, no-data value, pixel resolution, **zero-pixel count, no-data-pixel count, negative-pixel count** |

`image_attributes` is the expensive one — it opens and scans every single file.

Two quirks worth knowing:
- **Earth Engine sometimes returns more than one metadata record for the same image**, commonly with
  two different cloud-cover values. The handler de-duplicates, picking one arbitrarily, to guarantee
  one row per filename.
- The LLD admits the metadata file layout is awkward ("tech debt — we could've been a lot cleaner…
  the script which relationalizes these metadata has to navigate a strange file structure").

The CSVs go to S3, an **AWS Glue crawler** infers their schema, and they become tables queryable by
**AWS Athena** (SQL over files in S3 — no database server). See Part 5.

### Stage 4 — SQL filtering

**Where:** `sql/`

Two different questions, answered by near-identical queries:

**"Which glaciers should we work on?"** — `identify_inference_glims_ids_geog_area_rollup_50.sql`
ranks glaciers by `db_area` *within each of the five regions* (`dense_rank() over(partition by
geog_area_rollup order by db_area desc)`) and takes the top 50 per region. Ranking within region
rather than globally is deliberate: it stops the biggest-glacier regions from crowding everyone out.

**"Which individual images are usable?"** — `inference_data_query.sql` and
`training_data_query.sql`. Shared filter block:

```sql
where cloud_cover < 5
  and (image_quality = 9 or image_quality_oli = 9)
  and num_pixels > 50000
  and no_data_pixel_count = 0
  and percentage_zero_pixels < 0.1
  and geog_area != 'Canada'   -- excludes Ellesmere Island: too far north for DEM coverage
```

plus a **summer-months** flag computed per hemisphere:

```sql
case when is_southen_hemisphere = 0 and month_number between 5 and 10 then 1
     when is_southen_hemisphere = 1 and (month_number >= 11 or month_number <= 4) then 1 end
     as is_low_snow_month
```

Four things to know about this SQL, all verified by reading it:

1. **The cloud threshold here is `< 5`, but the LLD says 10** and the Earth Engine download filter
   uses `<= 10`. So the committed SQL is stricter than the documented intent. Probably a leftover
   from before the threshold was relaxed. Check before relying on either number.
2. **`num_pixels` is `height × width × num_of_bands`** — it counts band-multiplied cells, not
   ground pixels. So "more than 50,000 pixels" is really "more than ~6,250 ground pixels" for an
   8-band image. `percentage_zero_pixels` divides by the same inflated denominator. The LLD flags
   this itself: *"we should break down the 1,000,000 ways this is a dumb thing to do."* Don't
   reason about these two columns as if they were pixel counts.
3. **Hemisphere is approximated** as `geog_area_rollup in ('South America', 'Oceania')`. Coarse —
   it puts all of South America in the southern hemisphere — but fine for this glacier set.
4. **`rank_score = 1`** picks the single most recent qualifying image per glacier. It's *enabled*
   in `training_data_query.sql` (training wants one image per glacier) and deliberately
   **commented out** in `inference_data_query.sql` (inference wants the whole time series). That
   one commented line is the difference between "one snapshot" and "40-year series."

### Stage 5 — Build the training set

**Where:** `notebooks/pipeline/09_build_masks.ipynb`, then `get_training_set.ipynb`

`mask_creator.ipynb` turns each glacier's lat/long GLIMS polygon into a pixel mask aligned to that
glacier's imagery: build a glacier→CRS map from `image_attributes.csv`, reproject the polygon into
that CRS, then rasterize it (polygon in, 1s-inside/0s-outside raster out). Output goes to
`training/data/masks_staging_2/` at each glacier's **native** resolution — e.g. 50×57 pixels for one
small glacier (verified by reading the TIFF headers).

`get_training_set.ipynb` then assembles the trainable pairs: take the filtered file list from SQL,
select the common bands, normalize, resize everything to 128×128, stack the DEM on as a final
channel, and write matched `images/` and `masks/` tifs.

**One step the paper documents that's easy to miss: training images are temporal composites, not
single scenes.** All Landsat scenes within a **±100-day window** of the GLIMS `src_date` (the
outline's as-of date) were downloaded and combined by **arithmetic mean**, to average away shadows,
debris and transient defects. So "one training image per glacier" means one *averaged* image, built
from however many scenes happened to be available in that window — which varies per glacier.

The paper also describes reprojecting the **rasters** to WGS 84 to match the polygons, whereas
`mask_creator.ipynb` reprojects the **polygons** into each raster's UTM CRS. Same goal, opposite
direction. Probably an evolution in approach; don't assume either description matches how a given
artifact on disk was produced.

**⚠️ The channel count produced here does not match what `train.py` expects.** This is the single
most important thing to understand about this repo — full detail in Part 7.

### Stage 6 — Train

**Where:** `glacierview.training` (`dataset.py`, `losses.py`, `trainer.py`), driven by
`glacierview train`

A single 593-line script, no functions to speak of, runs top to bottom: build a CSV of
image/mask paths (dropping near-blank masks via `img.var() > 0.001`), 90/10 train/test split, a
`GlacierDataset` that normalizes per-image and computes NDSI/NDWI on the fly, random flips and
blurs for augmentation, then the U-Net, Adam, cross-entropy, and an evaluation printing mean Dice
and Jaccard. Saves to `experiments/<counter>/model` **and** overwrites
the configured output directory.

Hyperparameters are a mix of CLI flags (`--epochs --lr --decay --batch`) and module-level constants
you have to edit in the file (`LOSS`, `TRANSFORMS`, `NORMALIZE`, `UNFREEZE_WEIGHTS`, …). Note
`SHUFFLE_DATASET = False`, which is unusual for training.

⚠️ **The script's defaults are not the published recipe.** `--batch` defaults to 2; the paper used
**32**. The script also has no evaluation set and keeps whatever the last epoch produced, while the
paper selected the epoch that minimized cross-entropy on a held-out eval split. Full comparison table
in Part 7.

### Stage 7 — Inference

Two entry points that do overlapping things differently:

**`glacierview infer`** — one glacier, via `--glims-id`. Reads that glacier's
images, preprocesses, predicts, writes a GIF (every 5th frame) and a surface-area plot. Note it
plots `np.sqrt(np.sum(prediction))` and titles it "Estimated Surface Area (no units)" — that square
root is *not* an area, and this path is best read as a visual sanity check, not a measurement.

The paper's stated order of operations is: **sort by date → resize to 128×128 → Gaussian-smooth
along the time axis → compute NDSI/NDWI → CNN → resize back → count pixels.** That temporal smoothing
is `gaussian(X, sigma=[20,0,0,0], mode='reflect')` — σ = 20 *time points*, i.e. each pixel is
smoothed against itself at neighbouring dates to suppress per-scene noise. It is not spatial blur.

⚠️ The two scripts disagree about where smoothing sits, and the paper sides with `infer.py`:
both now smooth first and derive the indices from the smoothed stack, which is the published order.
They previously disagreed, because the sequence was written out twice.

**`glacierview areas`** — the batch job. Loops every glacier in the landing zone,
and for each one computes areas three ways (no threshold, thresholded, binarized), resizes
predictions **back to the image's original size** before counting, converts to km² with `* 0.0009`,
averages per year, and merges into three wide CSVs (one column per glacier). Also writes GIFs.
Each glacier is isolated: a failure is logged with a traceback, counted, and the run continues.
**Read the summary it prints** — and the exit code, which is non-zero if anything failed.

### Stage 8 — Analysis

**Where:** `notebooks/analysis/areas_true_vs_predicted.ipynb`

Compares the model's areas against GLIMS `db_area`, grouped by region. **Its four input CSVs were
missing and have now been recovered** — they live in `data/analysis/`; see Part 8 for what each
contains and the one caveat.

The statistical method, from the paper's appendix, is worth understanding because it's what turns
noisy per-date areas into the headline number. Glaciers differ hugely in size, so averaging areas in
km² would let the biggest glaciers dominate. Instead everything is done in **log space**:

$$\log(y_i) = \beta_0 + \beta_1 t_i + e_i$$

fitted by OLS per glacier. Because $\hat\beta_1$ is small, a first-order Taylor expansion makes it
read directly as **relative change per unit time** — so $\hat\beta_1 = -0.02$ means 2% area loss per
year. That's why every result is a relative rate rather than an absolute one.

Two estimators are reported: (1) fit the regression per glacier, then take a **weighted average** of
the slopes across a region, weighting by each glacier's number of valid observations; (2) take the
**geometric mean** across glaciers at each timestamp, then regress that single averaged series
(weighted, since each timestamp averages a different number of glaciers). Confidence intervals come
from a **bootstrap** — resample glaciers with replacement, refit, take the 2.5th/97.5th percentiles
(100 iterations for the regression variant, 500 for the slope variant).

⚠️ One thing to check in the code if you touch this: the paper's methods text gives the weight as
`(total_rows − valid_observations) / total_rows`, which is the proportion of data **missing** — yet
the surrounding sentence says it "gives greater influence to glaciers with more complete time
series." Those contradict each other. Either the formula or the description is wrong, and it affects
every weighted average in the results. Worth verifying against the actual regression script before
trusting or reproducing the weighting.

---

## Part 4 — Code tour: what every file is

The repository is one installable package plus notebooks. Everything that runs
the same way every time is in the package and driven from the CLI; notebooks
are for work that wants cell-by-cell iteration.

```
src/glacierview/        the package
notebooks/              pipeline / exploration / analysis
sql/                    Athena queries
data/                   committed reference data and a training fixture
docs/                   this guide, the design doc, the model manifest
figures/                diagrams and plots
.agents/                agent-facing rules, context, skills, references
```

### `src/glacierview/` — the package

| Module | Role |
|---|---|
| `config.py` | Every path and constant. Derived from `GV_REPO_ROOT` and `GV_DATA_ROOT`, both environment-overridable, so the ~110 GB of imagery need not sit in the checkout. **Run `glacierview config` to see everything resolved** — it is the fastest way to diagnose a path problem. |
| `log.py` | Logging setup. Named `log` so it does not shadow stdlib `logging`. |
| `rasters.py` | Reads GeoTIFFs and DEMs into HWC arrays, clamping negatives and `-inf` to zero. `get_rasters` takes an optional `filtered_file_path` — a CSV of filenames — which is how the metadata quality filter is applied before anything reaches memory. |
| `preprocess.py` | The shared pipeline: select common bands per satellite, normalize, resize to 128×128, stack the DEM, prepend NDSI and NDWI. Both inference paths use it, so they cannot drift. |
| `bands.py` | Pure data: per-satellite band-name→index maps. The `extra*` padding entries must stay — the selection mask is positional over the raster's band axis. |
| `viz.py` | Plotting helpers for eyeballing rasters in notebooks. |
| `models/unet.py` | **The only definition of the model.** `IN_CHANNELS = 9`. |
| `models/checkpoints.py` | `load_checkpoint` and `save_state_dict`. Isolates the pickle quirk described in Part 7. |
| `earthengine/export.py` | The `EePull` class: one method per Landsat satellite plus NASADEM, with server-side filtering. |
| `inference/predict.py` | The shared core: build inputs, predict, convert masks to km². |
| `inference/render.py` | GIFs and area plots. Uses a headless matplotlib backend. |
| `inference/batch.py` | The whole-landing-zone runner, with per-glacier failure isolation and a summary tally. |
| `training/dataset.py` | `build_manifest` (pairs images to masks on GLIMS ID) and `GlacierDataset`. |
| `training/losses.py` | Cross-entropy, Dice, and local `dice_score` / `jaccard_index` metrics. |
| `training/trainer.py` | `TrainConfig`, the epoch loop, eval-based model selection, scoring. |
| `cli.py` | `glacierview {config,infer,areas,train}`. |

### `notebooks/`

See `notebooks/README.md` for a per-notebook description. In short:

- **`pipeline/`** — 10 numbered steps that build the datasets. Kept as
  notebooks because each runs rarely and involves judgement about what to keep.
  They are drivers: the logic they call lives in the package.
- **`exploration/`** — 6 sandboxes. Not expected to run top to bottom.
  `inspect_training_data.ipynb` is the best first look at the actual data.
- **`analysis/`** — the published area analysis.

### `sql/` — Athena queries

| File | Role |
|---|---|
| `training_data_query.sql` | One representative summer image per glacier. |
| `inference_data_query.sql` | All qualifying summer images per glacier (`rank_score` commented out — that one line is the difference between a snapshot and a 40-year series). |
| `identify_inference_glims_ids_geog_area_rollup_50.sql` | Top 50 glaciers per region by `db_area`. |
| `denormalized_training_metadata.sql` | De-dupes `ee_metadata` by version, coalesces the two image-quality columns. |

**The 250-per-region selection query does not exist.** Its imagery and Athena
tables do. Adapt the `_50` query: `geog_size_rank <= 250`, pointed at the
`_250` tables.

### `data/`

| Path | Role |
|---|---|
| `analysis/` | Four committed CSVs the analysis notebook reads. See Part 8. |
| `sample/training/` | 32 image/mask pairs, so `glacierview train` can be smoke tested with no download. |
| `glims_18k.parquet` | The glacier table Athena reads. |

### Elsewhere

| Path | Role |
|---|---|
| `docs/low_level_design.md` | The original design doc. Read for rationale; its specifics have drifted. |
| `docs/MODEL_MANIFEST.md` | Per-checkpoint record: channel count, SHA-256, training recipe. |
| `AGENTS.md`, `.agents/` | Agent-facing guidance. `CLAUDE.md` is a symlink to `AGENTS.md`. |
| `figures/` | Diagrams and plots; `figures/README.md` says what each shows. **Look at these early.** |
| `qgis/` | A QGIS project — useful for eyeballing a GeoTIFF against a basemap. |
| `src/segmentation/gifs/` | ~150 output GIFs from previous runs. Watch a few; they make the project click. |


## Part 5 — The AWS account

Everything below was verified with the AWS CLI against the project AWS account. Run
`aws sts get-caller-identity` to confirm your credentials point at it.

### Two regions, and why it bites

Buckets are split between **`us-east-1`** (N. Virginia) and **`us-west-1`** (N. California). There's
no pattern to it — it's historical. Consequences:
- Every `aws s3` command needs the correct `--region`.
- The Athena/Glue catalog is in **us-west-1** but reads metadata buckets in **us-east-1**. That works
  fine; it's just cross-region.
- Pulling the big imagery bucket is a cross-region transfer if you're working in the east.

### The buckets that matter

| Bucket | Region | Contents (verified) | Where it belongs locally |
|---|---|---|---|
| `full-time-series-images-t1-l2-sr` | us-west-1 | **The inference imagery.** `{glims_id}/*.tif` plus `{glims_id}/meta_data/metadata_list_l{5,7,8}`. 1,101 glacier prefixes. **486 GB / 516,152 objects.** | `ee_landing_zone/full_time_series_c02_t1_l2/landsat/` |
| `full-time-series-dems-t1-l2-sr` | us-east-1 | Flat `{glims_id}_NASADEM.tif`. 1,122 objects / 89 MB. | `ee_landing_zone/full_time_series_c02_t1_l2/dems/` |
| `full-time-series-metadata-t1-l2-sr` | us-east-1 | One prefix per Glue table. Two vintages: `*_full_time_series/` and `*_full_time_series_250/`. | `processed_metadata/full_time_series_c02_t1_l2/` |
| `raw-training-images-t1-l2-sr` | us-west-1 | **Raw training imagery.** `landsat/{glims_id}/*.tif` + `logs/`. 96 GB / 303,233 objects, 17,945 glaciers. | `ee_landing_zone/localized_time_series_for_training_c02_t1_l2/landsat/` |
| `training-images-t1-l2-sr` | us-west-1 | **⚠️ Not raw imagery.** The *finished* training set: `images/` (7,174 tifs, 128×128×7) + `masks/` (8,902 tifs, 128×128×1). 7.2 GB. | `training/src/training_data/{images,masks}/` |
| `training-images-metadata-t1-l2-sr` | us-west-1 | The four training metadata tables + `glims_18k/glims_18k.parquet`. | `processed_metadata/localized_time_series_for_training_c02_t1_l2/` |
| `full-time-series-images-t1-l2-sr-trient-only`<br>`training-images-metadata-t1-l2-sr-trient-only` | us-west-1 | Single-glacier (Trient) copies. 1,753 objects / 755 MB. | Ideal smoke test. |
| `segmentation-model-gv` | us-east-1 | **Misleading name — contains no model.** Only `training_data.pickle` (1.7 GB), `test_data.pickle` (582 MB), and an older `training_data/{images,masks}` copy. 22.6 GB. | — |
| `athena-output-gv`, `glacier-view-athena-output-us-west-1` | east / west | Athena scratch — where query results land. Output only. | — |
| `mwaismann-gv` | us-west-1 | A `test/` prefix, 2 objects. Scratch. | — |

**The naming trap, stated plainly:** `training-images-t1-l2-sr` is the *processed, ready-to-train*
data. `raw-training-images-t1-l2-sr` is the *raw* data. The names suggest the opposite of the truth.

### Legacy buckets — archive, do not wire in

These have `-gv` suffixes and 2021–2022 dates, and predate the Collection-2 rework:
`joined-time-series-gv`, `joined-dems-gv`, `training-images-gv`, `training-pickles-gv`,
`training-metadata-gv`, `segmentation-metadata-gv`, `image-quality-classifier-gv`.

Two of these are historically interesting: `training-pickles-gv` holds the pre-GeoTIFF era when data
was shipped as pickled arrays, and `image-quality-classifier-gv` is the remains of a *separate
model* that classified image quality — `src/image_quality_classifier/` still appears in `.gitignore`
but the module no longer exists in the repo. That idea was abandoned in favour of metadata filtering.

### Athena and Glue

**Athena** runs SQL directly over CSV files in S3. **Glue** is the catalog: a crawler scans a prefix,
infers columns, registers a table. So "the database" is just S3 files plus schema metadata.

Catalog: **`glacier-view` in us-west-1**. Verified tables and their backing prefixes:

```
ee_metadata                            → s3://training-images-metadata-t1-l2-sr/ee_metadata/
file_attributes                        → s3://training-images-metadata-t1-l2-sr/file_attributes/
image_attributes                       → s3://training-images-metadata-t1-l2-sr/image_attributes/
processed_logs                         → s3://training-images-metadata-t1-l2-sr/processed_logs/
glims_18k                              → s3://training-images-metadata-t1-l2-sr/glims_18k/
ee_metadata_full_time_series           → s3://full-time-series-metadata-t1-l2-sr/ee_metadata_full_time_series/
file_attributes_full_time_series       → …/file_attributes_full_time_series/
image_attributes_full_time_series      → …/image_attributes_full_time_series/
processed_logs_full_time_series        → …/processed_logs_full_time_series/
ee_metadata_full_time_series_250       → …/ee_metadata_full_time_series_250/
file_attributes_full_time_series_250   → …/file_attributes_full_time_series_250/
image_attributes_full_time_series_250  → …/image_attributes_full_time_series_250/
processed_logs_full_time_series_250    → …/processed_logs_full_time_series_250/
```

Three vintages, and knowing which you're in is essential:
- **no suffix** = the training pull (one image per glacier)
- **`_full_time_series`** = the original 50-glaciers-per-region inference run
- **`_full_time_series_250`** = the September 2024 expansion to 250 per region

There are also `glacier-view-trient-only` and `test` databases in us-west-1, and `gv-metadata` /
`segmentation-metadata-gv` databases in us-east-1 (legacy).

⚠️ **The LLD's table names are wrong.** It lists things like
`file_attributesee_metadata_full_time_series_250` — a copy-paste mangling. The real names are the
clean ones above. The committed SQL uses unqualified names, so set your Athena database to
`glacier-view` and it resolves.

### The 250-glacier expansion is half-finished

This matters for planning. In September 2024 the project expanded from 50 to 250 glaciers per region:
- ✅ The imagery was downloaded — the bucket holds 1,101 glaciers.
- ✅ The metadata was processed and crawled — the `_250` tables exist and are populated.
- ✅ The glacier list exists — `geog_area_rollup_250.csv`, **1,024 unique glaciers**. (Not 1,250:
  Caucausus only yielded 74 and Europe 199 before running out of qualifying glaciers; Asia 250,
  South America 250, North America 251.)
- ❌ **The per-file filtering was never done for it.** The SQL file is empty (0 bytes), and the only
  filtered file list on disk — `filtered_inference_data.csv` — covers **245 glaciers** (50 per
  region, South America 45), i.e. the *old* 50 vintage.

So the expansion's data is sitting in S3, catalogued, waiting for a filter query that was never
written. Reconstructing it means adapting
`identify_inference_glims_ids_geog_area_rollup_50.sql` — change `geog_size_rank <= 50` to `<= 250`
and point it at the `_250` tables.

---

## Part 6 — The data model

### `data_label`: the switch that selects a dataset

Every script and notebook has a hardcoded string near the top:

```python
data_label = "full_time_series_c02_t1_l2"
```

That string picks which subdirectory of the landing zone to operate on. There is no config file —
changing dataset means editing this constant in each place. The live values:

| `data_label` | What it is |
|---|---|
| `localized_time_series_for_training_c02_t1_l2` | Training data: small crops, one image per glacier |
| `full_time_series_c02_t1_l2` | Inference data: full 1984–present series per glacier |
| `full_time_series` | ⚠️ Legacy, pre-Collection-2. **Empty on disk.** |
| `localized_time_series_for_segmentation_training` | Legacy |
| `localized_time_series_for_segmentation_training_large` | Legacy |

### Directory layout

```
src/earth_engine/data/
  ee_landing_zone/<data_label>/
    landsat/<glims_id>/<glims_id>_<date>_<L5|L7|L8>_C02_T1_L2_SR.tif
    landsat/<glims_id>/meta_data/metadata_list_l{5,7,8}    # raw EE JSON blobs
    dems/<glims_id>_NASADEM.tif
    logs/log_<name>.log
  processed_metadata/<data_label>/*.csv                     # the relationalized tables

src/glims/data/
  glims_db_20210914/glims_polygons.shp     # the full GLIMS inventory (source of truth)
  training_samples/glims_18k_bb.shp        # 18k glaciers + bounding boxes
  inference_samples/geog_area_rollup_{50,250}.csv
  joined_time_series/

src/segmentation/
  training/data/masks_staging_2/                        # native-resolution masks
  training/data/processed_training_data_summer_months/   # the actual training pairs
    images/  masks/
  training/src/training_data/{images,masks}   # ← where train.py looks (relative to its cwd)
  unet_summer_model_unfrozen_100              # the production checkpoint (~340 MB)
  gifs/  tmp/  surface_area_time_series/      # outputs
```

**None of `data/` is in git.** `.gitignore` excludes every data directory plus `*.csv`, `*.png`,
`*.gif`, `*.jpg`. A fresh clone has code and nothing else.

### Filenames are load-bearing

This is the single most important convention in the codebase:

```
G007026E45991N_1984-04-09_L5_C02_T1_L2_SR.tif
└──── 0 ─────┘ └─── 1 ───┘ └2┘
   glims_id       date    satellite
```

Split on `_`:
- **token 0** → the GLIMS ID
- **token 1** → the date, parsed with `datetime.strptime(..., '%Y-%m-%d')` in `infer.py:67` and
  the batch runner
- **token 2** → the satellite, lowercased and looked up in `glacierview.bands` by
  `glacierview.preprocess.get_common_bands()`

**Rename a file and the pipeline silently misreads its bands or its date.** Any new data must follow
this exact pattern.

### The preprocessing chain

Every entry point runs the same four steps in the same order:

1. **`read.get_rasters(dir)`** — opens every `.tif`, sets `-inf` and all negative values to 0,
   moves the band axis last (HWC), returns `{filename: array}`.
2. **`preprocess.get_common_bands(rasters, common_bands)`** — per satellite, select the named bands.
   It uses `np.isin(before_bands, common_bands)` as a boolean mask, which **preserves the original
   band order** rather than the order you listed. So the output order is the satellite's native
   order, filtered — and the point is that this comes out *consistent* across L5/L7/L8.
3. **`preprocess.normalize_rasters(rasters)`** — ⚠️ **not a plain min-max.** It finds the *second*
   smallest and *second* largest unique values, replaces the actual extremes with those, then scales
   to [0,1]. The reasoning: true min/max are usually sensor artifacts (dead pixels, saturation) and
   would squash the real signal. Worth knowing, because it means values slightly outside [0,1] can
   survive, and a uniform band is special-cased.
4. **`preprocess.resize_rasters(rasters, (128,128))`** — torchvision resize.

Then NDSI and NDWI are computed and prepended, and the DEM is concatenated. Details in Part 7.

The LLD flags a subtlety here: *"We should pick to resize based on a single size otherwise we get two
lines appearing in the line plot."* Inconsistent resize targets across a time series produce a
visible discontinuity in the area plot — an artifact of preprocessing, not real glacier change.

---

## Part 7 — The model contract

### Architecture

A U-Net with a ResNet-50 encoder, in PyTorch, defined once in
`glacierview/models/unet.py`.

```
input (N, 9, 128, 128)
   │
   ├─ ResNet-50 encoder, first conv replaced:
   │     nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
   │  (ImageNet weights kept for later layers; last two ResNet layers removed)
   │  → 2048 channels at 4×4
   │
   └─ decoder: 5 × ConvTranspose2d + BatchNorm + conv_block,
        skip connections from encoder stages x2/x4/x5/x6,
        raw input concatenated at the last level
   │
   └─ classifier: Conv2d(16, 2, kernel_size=1) → 2-class logits
```

The two output channels are glacier and background, one the complement of the
other; two rather than one only because PyTorch's cross-entropy expects
`num_classes` outputs.

Glacier probability is `softmax(logits, dim=1)[:, 1]`, thresholded at 0.5.
Area is `mask.sum() * 0.0009` km², computed **after** resizing back to the
scene's native resolution.

### Nine input channels

```
0: NDWI   1: NDSI   2: blue   3: green   4: red   5: nir   6: swir   7: thermal   8: DEM
```

`swir_2` is excluded deliberately — the paper's channel figure says so
("Remove SWIR2 because of quality") and its caption reads "The 9 channels used
in the CNN". The count is declared once, as `IN_CHANNELS` in
`glacierview/models/unet.py`, and derived everywhere else:

| Source | Channels |
|---|---|
| `config.COMMON_BANDS` | 6 bands + DEM = 7 → +NDSI +NDWI = **9** |
| training data on disk | 7-channel tifs → **9** |
| `IN_CHANNELS` | **9** |

Every entry point asserts its built tensor matches before the first
convolution, so a mismatch names the offending file instead of failing inside
a bare `except`.

*History worth knowing:* three source files once built 10 channels against
these 9-channel checkpoints, so inference failed on every glacier — silently,
because of that bare `except`. The paper's body text still says 10 where its
own figure says 9. The figure is right, and the checkpoints settle it: their
stem convolution is `Conv2d(9,64,7,7)`.

### Why checkpoints are awkward, and how that is handled

Checkpoints are `torch.save(model)` output — a pickled module, not a state
dict. The pickle records where the class came from, and the released
checkpoints were written by a script, so they reference `__main__.UNet` and
`__main__.conv_block`. Loading one from anywhere else used to raise:

```
AttributeError: Can't get attribute 'UNet' on <module '__main__'>
```

`glacierview.models.load_checkpoint` injects those names before unpickling, so
callers no longer import the classes for their side effect:

```python
from glacierview.models import load_checkpoint
model = load_checkpoint("path/to/checkpoint", device="cpu")
```

It also warns if the checkpoint's channel count differs from `IN_CHANNELS`.

Consequences that remain:

- Renaming or reshaping a layer can stop an existing checkpoint loading.
- `save_state_dict` is the right export for anything shared outside this repo:
  a state dict carries no class references and no code-execution risk.

### The two checkpoints

| File | Date | Size | Channels |
|---|---|---|---|
| `unet_summer_model_unfrozen_100` | 2024-10-02 | 343,522,287 B | 9 |
| `saved_models/model` | 2023-12-27 | 343,487,293 B | 9 |

Different weights, ~10 months apart, and nothing records which produced the
paper's 0.92 Dice. Both plus 18 legacy Keras `.h5` files are backed up to
`s3://segmentation-model-gv/checkpoints/` with a `MANIFEST.md`. Bucket
versioning there is still off, so an overwrite is unrecoverable.

### Training recipe

The CLI defaults **are** the published recipe:

| | Value |
|---|---|
| learning rate | 0.00001 |
| batch size | 32 |
| encoder | unfrozen |
| loss | cross-entropy |
| threshold | 0.5 |
| optimizer | Adam |
| split | 90 / 5 / 5 |
| model selection | epoch minimising eval loss |

Augmentation: vertical flip, horizontal flip, and a 3×3 Gaussian blur
(σ = 0.8), each applied **independently at p = 0.5** — about 75% more data,
not the 4× you would get applying all three every time.

Reported result: **Dice 0.92**. The ablation trail from the paper:

| Change | Dice |
|---|---|
| 3,000 images, no transfer learning | 0.83 |
| + transfer learning (ImageNet ResNet-50) | 0.86 |
| + NDSI & NDWI channels | 0.88 |
| + full dataset, cross-entropy | **0.92** |
| (Dice loss instead) | 0.90 |
| frozen encoder | 0.91 |
| ResNet-18 instead of ResNet-50 | 1–2% worse |

Threshold sweep: 0.1 → 0.16, 0.4 → 0.923, **0.5 → 0.926**, 0.8 → 0.87. So 0.5
is empirically optimal, not merely conventional.

⚠️ **Two bugs in the original training loop are fixed, and both change
behaviour.** `optimizer.zero_grad()` was never called, so gradients
accumulated across an entire epoch; and only the last batch's loss was
recorded per epoch. The published checkpoint was trained *with* those bugs, so
reproducing its exact numbers needs the pre-refactor script from git history.

---

## Part 8 — Data state, and what is still open

### Present and working

- **The package** — installs with `uv sync`, all submodules import, all CLI
  subcommands run, a released checkpoint loads and runs a forward pass.
- **Training data** — 7,174 image/mask pairs. `build_manifest` yields
  6,453 train / 359 eval / 359 test, matching the paper's 6,456/359/359.
- **A training fixture** — `data/sample/training`, 32 pairs, committed. Enough
  to smoke test training with no download.
- **Inference DEMs** — all 1,122, matching the bucket's ID set exactly.
- **Inference metadata** — the four table CSVs plus `filtered_inference_data.csv`.
- **Analysis inputs** — the four CSVs in `data/analysis/`:

| File | Shape | What it is |
|---|---|---|
| `geo_areas.csv` | 18,093 × 32 | The `glims_18k` table: `glac_id`, `db_area`, `geog_area`, `geog_area_rollup`, `bboxes`, elevations. The join key for regional grouping. |
| `training_data_set.csv` | 10,447 × 12 | Output of `training_data_query.sql` — candidate training images with quality metadata. This is the paper's stale "10,443". |
| `training_areas.csv` | 3 × 260 | A transposed lookup: ~259 glacier IDs → areas. |
| `areas_binary_05_filtered_smooth_25.csv` | 541 dates × **245** glaciers | Smoothed batch output, wide format. |

### The one blocking gap

**Inference imagery is absent locally.** `landsat/` under the inference
landing zone is empty; that tree lived on an external SSD. Nothing is missing
*upstream* — every glacier needed exists in S3. Refilling is 23.7 GB for the
filtered set, not the bucket's 486 GB. See Part 9.

### Still open

1. **The 250-per-region selection query** was never committed, though its
   imagery and Athena tables exist.
2. **Which checkpoint produced the 0.92 Dice** is unrecorded.
4. **Is the GLAMOS comparison still current?** A 14% underestimate on Trient
   in 2006 belongs in the paper's limitations if it holds.
5. **Is there any backup of `glims_db_20210914/`,** the dated GLIMS snapshot
   the entire glacier selection derives from? It appears to exist on one
   laptop only.

### Backed up, and not

Backed up to S3: both checkpoints, the legacy Keras files, all imagery and
metadata, the derived Parquet.

**Not backed up anywhere but one laptop:** the ~18,093 training-landing-zone
DEMs (there is no training DEM bucket), 148 raw training glaciers that are
local-only, the derived metadata CSVs, and `src/glims/data/glims_db_20210914/`
— the dated GLIMS snapshot the whole selection derives from.

---

## Part 9 — Getting set up

### Step 0 — install

```bash
uv sync
```

Reads `pyproject.toml`, installs the versions pinned in `uv.lock` into
`.venv/`, and downloads CPython 3.12 if absent (pinned in `.python-version`).
Then either prefix with `uv run` or activate `.venv`.

```bash
uv run glacierview --help
uv run jupyter lab          # notebook deps install by default
```

`requires-python` is capped at `<3.13`: uncapped, uv resolves against 3.14,
where rasterio, geopandas and torch have no wheels. If you widen it, re-run
`uv lock` **and** `uv sync`.

PyPI's torch gives a CPU/MPS build on macOS and a bundled-CUDA build on Linux.
The published model was trained against cu118; `pyproject.toml` carries a
commented `[[tool.uv.index]]` block for pinning that exactly.

### Step 1 — check the paths

```bash
uv run glacierview config
```

Prints every resolved path and marks what is missing. Nothing is hardcoded to
one machine: point the data root anywhere, for instance at an external disk.

```bash
export GV_DATA_ROOT=/Volumes/T7/GlacierView
```

### Step 2 — credentials

```bash
aws sts get-caller-identity     # confirm you are in the project AWS account
```

Earth Engine is only needed to pull *new* imagery: `ee.Authenticate()` once,
then `ee.Initialize()`.

### Step 3 — smoke test without downloading anything

```bash
uv run glacierview train --data-dir data/sample/training --epochs 2 \
  --batch-size 4 --out-dir /tmp/smoke --device cpu
```

Two epochs on 28 images. The Dice will be near zero — that is expected from
scratch — but the loss should fall, which proves the whole path works.

### Step 4 — one glacier, end to end

Trient (`G007026E45991N`) is the documented case, ~700 MB:

```bash
LZ=$(uv run glacierview config | awk '/landsat_dir/{print $2}')
aws s3 cp s3://full-time-series-images-t1-l2-sr/G007026E45991N/ \
  "$LZ/G007026E45991N/" --recursive --region us-west-1 \
  --exclude "*/meta_data/*" --exclude "*.DS_Store"

uv run glacierview infer --glims-id G007026E45991N \
  --checkpoint src/segmentation/unet_summer_model_unfrozen_100
```

Success is a GIF showing the ice boundary moving over 40 years. Compare
against the ~150 GIFs already in `src/segmentation/gifs/`.

The paper documents Trient in detail: 464 images, 1984-04-18 to 2023-10-07,
decline concentrated in 1985–1988 and 2002–2015, roughly stable over the last
decade. Two caveats — the paper's Trient run used looser filters (cloud < 20,
zero pixels < 15%), so image counts will not match unless you match the
filters; and the Swiss GLAMOS inventory measured Trient at 5.76 km² in 2006
against the model's 4.95 km², a **14% underestimate**. Expect a level offset
even when the trend is right.

### Step 5 — scaling up without pulling 486 GB

`filtered_inference_data.csv` narrows inference to **245 glaciers / 14,952
files / 23.7 GB**. `glacierview.rasters.get_rasters` accepts a
`filtered_file_path`, so the quality filter can be applied at read time rather
than relying on what you happened to download.

`aws s3 sync` was too slow for the original upload; `cp --recursive` is what
was used. Always exclude `.DS_Store` (some are 660 KB) and `meta_data/`.

### Step 6 — full training run

```bash
uv run glacierview train --epochs 10 --batch-size 32 --out-dir experiments/run1
```

`--data-dir` defaults to `config.training_data_dir()`. Each run writes
`model.state_dict.pt` and `run.json` (config, per-epoch train and eval losses,
best epoch, test Dice and Jaccard).

### What still does not exist

No test suite, no CI, no Docker, no orchestration. Dependencies are managed
and linting is set up (`uv run ruff check .` passes; `ruff format` is
deliberately not enforced). Adding a small pytest suite over the pure
functions — `dice_score`, `areas_km2`, `_to_chw`, `normalize_rasters`,
`build_manifest` — would be the highest-value next step.

---

## Part 10 — Landmines, ranked

Ordered by how likely they are to cost you a day.

1. **Inference failures are logged, not raised.** `glacierview areas` isolates
   each glacier so one bad scene does not kill a run over hundreds. **Read the
   summary it prints** and check the exit code — a run can complete having
   failed on most glaciers.
2. **`num_pixels` and `percentage_zero_pixels` are band-inflated**
   (`height × width × num_of_bands`). "More than 50,000 pixels" is really
   ~6,250 ground pixels for an 8-band image. The design doc calls this out as
   a known mistake. Do not reason about them as ground-pixel counts.
3. **Cloud cover is filtered twice, at different thresholds.** `<= 10`
   server-side at download (`earthengine/export.py`), then **`< 5`** in the
   pre-inference SQL. Verified against the data: both
   `filtered_inference_data.csv` and `filtered_training_data_summer_months.csv`
   top out at 4.99 and 4.34 with zero rows at or above 5, so `< 5` is what
   produced the published results. The design doc's note about relaxing the
   threshold to 10 only ever reached the download step. Consequence worth
   knowing: **there is downloaded imagery between 5 and 10 cloud cover that
   inference has never used.** Relaxing the SQL would add images with no new
   download, but would diverge from the paper. The paper's Trient case study
   used a looser `< 20` as a deliberate one-off.
4. **Images and masks pair on the GLIMS ID prefix, not on filename.** Images
   are named per scene, masks per glacier. Assuming matching names finds
   nothing — which is exactly what the old training script did.
5. **Two training-tif vintages exist**, `(1,128,128,7)` and `(128,128,8)`. The
   loader normalises both, but anything reading tifs directly should not
   assume a shape.
6. **Checkpoints are pickled modules.** Use `load_checkpoint`; renaming a
   layer can break an old checkpoint.
7. **Reproducing the paper needs the pre-refactor training script** — the
   `zero_grad` and loss-recording fixes change training dynamics.
8. **`torchmetrics` renames metrics between minor versions.** `Dice` vanished
   in 1.9. Dice and Jaccard are local functions now; keep them that way.
9. **A stray `full_time_series_c02_t1_l2/G007026E45991N/`** sits beside
   `landsat/` rather than inside it — an old download one level too high.
10. **Notebooks hardcode nothing now, but still share one namespace.** A
    variable that looks unused in its cell may be read by a later one, which
    is why several ruff rules are exempted for `*.ipynb`.
11. **Region flags.** Buckets are split across `us-east-1` and `us-west-1`;
    forget `--region` and `aws s3` fails confusingly.
12. **`.DS_Store` files are in S3**, some 660 KB, mixed in with the tifs.


## Part 11 — Glossary

| Term | Meaning |
|---|---|
| **Athena** | AWS service running SQL directly over files in S3 |
| **band** | one wavelength channel of a satellite image |
| **C02 / T1 / L2** | Collection 2 / Tier 1 / Level 2 (surface reflectance) — the Landsat product tier used here |
| **CRS** | Coordinate Reference System — how pixel positions map to Earth positions |
| **`data_label`** | hardcoded string selecting which dataset directory to use |
| **DEM** | Digital Elevation Model — raster of ground elevations |
| **Dice / Jaccard (IoU)** | overlap metrics between predicted and true masks, 0–1 |
| **EPSG code** | numeric identifier for a CRS (e.g. 4326 = lat/long, 326xx = UTM) |
| **GEE** | Google Earth Engine — hosted planetary imagery archive |
| **GeoTIFF** | TIFF image carrying geographic metadata |
| **GLIMS** | Global Land Ice Measurements from Space — the glacier inventory used here |
| **GLIMS ID** | `G<lon>E<lat>N` glacier identifier, e.g. `G007026E45991N` |
| **Glue** | AWS service that crawls S3 files and registers table schemas |
| **`geog_area`** | GLIMS free-text region field |
| **`geog_area_rollup`** | this project's 5-way region bucketing: Asia, Europe, North America, South America, Caucausus |
| **`db_area`** | GLIMS-published glacier area in km², used as ground truth |
| **landing zone** | the local directory tree holding downloaded imagery |
| **mask** | per-pixel label image; here 1 = glacier, 0 = not |
| **NASADEM** | the specific global DEM product used |
| **NDSI** | Normalized Difference Snow Index = `(green−swir)/(green+swir)` — high for ice |
| **NDWI** | Normalized Difference Water Index = `(green−nir)/(green+nir)` — high for water |
| **nir / swir** | near infrared / shortwave infrared |
| **rasterio** | Python library for reading GeoTIFFs |
| **RGI** | Randolph Glacier Inventory — a different inventory, deliberately excluded |
| **ResNet-50** | 50-layer CNN used here as the U-Net encoder, ImageNet-pretrained |
| **semantic segmentation** | classifying every pixel of an image |
| **skip connection** | wire from encoder to matching decoder level, preserves fine detail |
| **SR** | Surface Reflectance |
| **U-Net** | encoder–decoder segmentation architecture with skip connections |
| **UTM zone** | one of 60 north–south strips, each with its own local flat projection |

---

## Appendix — Where the truth lives

When sources disagree, this is the precedence order:

1. **The checkpoints and the data on disk.** Byte-level facts beat every document, and where this
   guide makes a claim about channels, shapes or counts it was checked against them.
2. **The paper.** Authoritative on the *intended* design and on every published number: the 9
   channels, the training recipe, the results. Its figures are more current than its body text, and
   it carries two stale numbers of its own (noted in Part 1).
3. **This document** and `.agents/` — written from reading the code, querying AWS, and reading the
   paper. `.agents/` is the agent-facing source of truth; this document is the long-form narrative
   it points back to.
4. **The code.** Authoritative on what *currently runs*, which in several places is not the intended
   design. Since the refactor the two agree on channel count, which was not true before.
5. **`low_level_design.md`** — authoritative on *rationale* (why thresholds, what the tech debt is,
   what the risks were). Its *specifics* have drifted: Glue table names are mangled, the band order
   is listed NDSI-first when the code prepends NDWI last, and its cloud-cover figure describes the
   download filter rather than the stricter pre-inference one.
6. **`README.md`** — the oldest of these (Nov 2023). Its directory layout
   (the training output directory, `inference/src`) partly predates the current tree, and it
   describes the model as taking "128x128 pixels with 7 bands" — which is neither the 8-channel
   design nor the 10-channel model input. Treat as historical.

### Open questions

**Answered by the paper** (previously open in this doc):

- ~~Which channels was the model trained on?~~ **Nine**: NDWI, NDSI, blue, green, red, nir, swir,
  thermal, DEM. `swir_2` was dropped for quality. Confirmed directly against both checkpoints.
- ~~Where are the four analysis CSVs?~~ Recovered, now in `data/analysis/`.
- ~~Where is the 1,083-glacier areas output?~~ **Recovered**:
  `data/analysis/areas_binary_05_filtered_smooth_f.xlsx`, 1,085 glacier columns. Replicating the
  notebook's annual-mean grouping and regressing per region gives exactly the paper's 1,083.
- ~~Cloud-cover threshold: 5 or 10?~~ **Both, at different stages.** `<= 10` server-side at
  download, `< 5` in the pre-inference SQL. Verified against the filtered file lists, which top out
  at 4.99 and 4.34 with no rows at or above 5. The design doc's relaxation to 10 only reached the
  download step.
- ~~Is the regression weighting correct?~~ **No — it is inverted.** The code weights by
  `(total_rows - n_observations) / total_rows`, the proportion of data *missing*. Corrected, the
  overall rate moves from −0.06706 to −0.07557 and North America stops being an outlier. See
  `.agents/context/state.md`.
- ~~Where did the 35 Oceania glaciers go?~~ They never entered inference: **zero** of them have a
  column in the areas file, and both analysis loops iterate `areas[:-1]`, which slices Oceania off
  the end of the region list.

**Still open** — these need Somansh Budhwar or Armin Schwartzman (`armins@ucsd.edu`):

1. **Does the 250-glacier selection SQL exist anywhere?** The committed file was a 0-byte
   placeholder, yet 1,101 glaciers were downloaded and the `_250` Athena tables are populated. The
   query that chose them is unrecorded.
2. **What distinguishes the two PyTorch checkpoints?** `saved_models/model` (Dec 2023) vs
   `unet_summer_model_unfrozen_100` (Oct 2024). Same architecture, different weights, no record of
   which produced the paper's 0.92 Dice.
3. **Is the GLAMOS comparison still current?** A commented-out passage notes Trient measured
   5.76 km² in 2006 against the model's 4.95 km² — a 14% underestimate. If that holds, it belongs in
   the paper's limitations.
