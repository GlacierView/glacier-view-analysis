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
7. [The model contract (and the channel bug)](#part-7--the-model-contract-and-the-channel-bug)
8. [What's missing, broken, or unbacked](#part-8--whats-missing-broken-or-unbacked)
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
`src/segmentation/helpers/landsat_bands.py` is the lookup table that fixes this, and
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

**Where:** `src/glims/src/glims_processor.ipynb`, then `glims_to_parquet.ipynb`

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

**Where:** `src/earth_engine/src/gee_readers/ee_helpers.py` + the `ee_pull_*.ipynb` notebooks

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

**Where:** `src/earth_engine/src/metadata/time_series_metadata_handler.ipynb`
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

**Where:** `src/sql/`

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

**Where:** `src/segmentation/mask_creator.ipynb`, then `get_training_set.ipynb`

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

**Where:** `src/segmentation/training/src/train.py`

A single 593-line script, no functions to speak of, runs top to bottom: build a CSV of
image/mask paths (dropping near-blank masks via `img.var() > 0.001`), 90/10 train/test split, a
`GlacierDataset` that normalizes per-image and computes NDSI/NDWI on the fly, random flips and
blurs for augmentation, then the U-Net, Adam, cross-entropy, and an evaluation printing mean Dice
and Jaccard. Saves to `experiments/<counter>/model` **and** overwrites
`src/segmentation/inference/model`.

Hyperparameters are a mix of CLI flags (`--epochs --lr --decay --batch`) and module-level constants
you have to edit in the file (`LOSS`, `TRANSFORMS`, `NORMALIZE`, `UNFREEZE_WEIGHTS`, …). Note
`SHUFFLE_DATASET = False`, which is unusual for training.

⚠️ **The script's defaults are not the published recipe.** `--batch` defaults to 2; the paper used
**32**. The script also has no evaluation set and keeps whatever the last epoch produced, while the
paper selected the epoch that minimized cross-entropy on a held-out eval split. Full comparison table
in Part 7.

### Stage 7 — Inference

Two entry points that do overlapping things differently:

**`src/segmentation/inference/infer.py`** — one glacier, via `--glimsid`. Reads that glacier's
images, preprocesses, predicts, writes a GIF (every 5th frame) and a surface-area plot. Note it
plots `np.sqrt(np.sum(prediction))` and titles it "Estimated Surface Area (no units)" — that square
root is *not* an area, and this path is best read as a visual sanity check, not a measurement.

The paper's stated order of operations is: **sort by date → resize to 128×128 → Gaussian-smooth
along the time axis → compute NDSI/NDWI → CNN → resize back → count pixels.** That temporal smoothing
is `gaussian(X, sigma=[20,0,0,0], mode='reflect')` — σ = 20 *time points*, i.e. each pixel is
smoothed against itself at neighbouring dates to suppress per-scene noise. It is not spatial blur.

⚠️ The two scripts disagree about where smoothing sits, and the paper sides with `infer.py`:
`infer.py` smooths first and derives the indices from the smoothed stack; `final_areas.py` computes
the indices first and then smooths everything including them.

**`src/segmentation/final_areas.py`** — the real batch job. Loops every glacier in the landing zone,
and for each one computes areas three ways (no threshold, thresholded, binarized), resizes
predictions **back to the image's original size** before counting, converts to km² with `* 0.0009`,
averages per year, and merges into three wide CSVs (one column per glacier). Also writes GIFs.
Every glacier is wrapped in `try: … except: print(f"Error {glims_id}")` — so a run can appear to
succeed while silently failing on most glaciers. **Always check the count of `Error` lines.**

### Stage 8 — Analysis

**Where:** `src/segmentation/areas_true_vs_predicted_final.ipynb`

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

### `src/glims/` — glacier selection

| File | Role |
|---|---|
| `src/glims_processor.ipynb` | **Live.** Stage 1. GLIMS → filtered set + bounding boxes. Also generates the figures in `figures/`. |
| `src/glims_to_parquet.ipynb` | **Live.** Adds `geog_area_rollup` (the region mapping dict lives here), drops geometry, writes `glims_18k.parquet` for Athena. |
| `src/glims_exploratory_data_analysis.ipynb` | Exploration. Region/locality summaries and plots. |

### `src/earth_engine/` — data acquisition

**Note: this whole directory is untracked in git.** It exists on disk but was never committed.

| File | Role |
|---|---|
| `src/gee_readers/ee_helpers.py` | **Live, the core.** The `EePull` class. Only real module here. |
| `src/gee_readers/ee_pull_time_series.ipynb` | **Live.** Drives `EePull` for the full inference time series. |
| `src/gee_readers/ee_pull_training.ipynb` | **Live.** Same, for the one-image-per-glacier training pull. |
| `src/gee_readers/ee_pull_dems.ipynb` | **Live.** Pulls one NASADEM per glacier. |
| `src/gee_readers/ee_list_of_all_files_pulled.ipynb` | Utility — inventories what was downloaded. |
| `src/gee_readers/get_s3_data.ipynb` | Utility — S3 fetch helper. |
| `src/metadata/time_series_metadata_handler.ipynb` | **Live.** Stage 3 for inference data. |
| `src/metadata/training_metadata_processor.ipynb` | **Live.** Stage 3 for training data. |
| `src/metadata/metadata_exploratory_data_analysis.ipynb` | Exploration — this is where the filter thresholds were chosen from real distributions. Read it to understand *why* the numbers are what they are. |

### `src/sql/` — Athena queries

| File | Role |
|---|---|
| `training_data_query.sql` | **Live.** One representative summer image per glacier. |
| `inference_data_query.sql` | **Live.** All qualifying summer images per glacier (`rank_score` commented out). Produced `filtered_inference_data.csv`. |
| `identify_inference_glims_ids_geog_area_rollup_50.sql` | **Live.** Top 50 glaciers per region. |
| *(none)* | **The 250-per-region selection query does not exist.** An empty 0-byte placeholder for it was deleted, since it read as a real query. Adapt the `_50` query above: change `geog_size_rank <= 50` to `<= 250` and point it at the `_250` tables. |
| `denormalized_training_metadata.sql` | Helper — de-dupes `ee_metadata` by version per file, coalesces `image_quality`/`image_quality_oli`. |

### `src/segmentation/` — the model and inference

**Shared helper modules** (`helpers/`) — imported by nearly everything:

| File | Role |
|---|---|
| `read.py` | **Live.** `get_rasters()` (dir → `{filename: HWC array}` dict, clamping `-inf` and negatives to 0), `get_dem()`, `reproject_raster()`. |
| `preprocess.py` | **Live.** `get_common_bands()`, `normalize_rasters()`, `resize_rasters()`. The heart of cross-satellite normalization. |
| `landsat_bands.py` | **Live.** Pure data: the per-satellite band-name→index table. |
| `explore.py` | Plotting helpers for eyeballing rasters in notebooks. |
| `model.py` | **DEAD.** TensorFlow/Keras from a previous iteration. Don't extend. |

**Scripts:**

| File | Role |
|---|---|
| `final_areas.py` | **Live.** Stage 7 batch inference. The main production entry point. |
| `inference/infer.py` | **Live.** Single-glacier inference + GIF. Contains its own inlined copy of the U-Net. |
| `inference/cnn.py` | **Live.** The U-Net class that `final_areas.py` imports. Also contains unused experimental variants: `AUNet`, `AUNet_NR`, `AttentionGate`, `Recurrent_block`, `RRCNN_block` (attention and recurrent U-Nets that were tried). Defines `conv_block` **twice** (lines 7 and 207) — verified byte-identical, so harmless, but confusing. |
| `inference/model.py` | **DEAD.** Another Keras copy. |
| `training/src/train.py` | **Live.** Stage 6. |

**Notebooks:**

| File | Role |
|---|---|
| `mask_creator.ipynb` | **Live.** Stage 5a — polygons → masks, handles the UTM problem. |
| `get_training_set.ipynb` | **Live.** Stage 5b — assembles image/mask pairs. |
| `areas_true_vs_predicted_final.ipynb` | **Live but unrunnable** — inputs missing. |
| `image_preprocessing_dev.ipynb` | Sandbox for the preprocessing functions. Good place to learn what the helpers do. |
| `band_distributions.ipynb` | Exploration of per-band pixel-value distributions. |
| `inspect_processed_training_data.ipynb` | Visual QA of the built training set. **Run this first** to see what the data actually looks like. |
| `reprojection.ipynb` | Reprojection experiments. |
| `test_saved_model_matt.ipynb`, `test_saved_model_somansh.ipynb` | Two people's model-testing notebooks. Useful as worked examples of loading a checkpoint and predicting. |
| `glacier_segmentation-updated.ipynb` | An older end-to-end train+predict notebook, superseded by `train.py`. |

### Elsewhere

| Path | Role |
|---|---|
| `low_level_design.md` | The LLD. Read for rationale; verify its specifics. |
| `AGENTS.md`, `.agents/` | Agent-facing guidance, split into rules (how to behave), context (what to know), skills (how to do a task) and references. `CLAUDE.md` is a symlink to `AGENTS.md`, so Claude Code and Codex read the same entrypoint. |
| `figures/` | Diagrams and plots, including `data_collection_and_preprocessing.drawio` (editable pipeline diagram) and the UTM-overlap illustrations. **Look at these early** — they explain the pipeline faster than prose. |
| `src/styles/ieee.mplstyle` | Shared matplotlib style for publication-ready plots. |
| `figures/inventory-examples/` | Screenshots showing why the RGI and GlobGlacier outlines were excluded. |
| `qgis/` | A QGIS project file — QGIS is the standard desktop GIS tool, useful for eyeballing a GeoTIFF against a basemap. |
| `figures/cnn_architecture.png` | The U-Net architecture diagram, shown in the README. |
| `src/segmentation/gifs/` | Output GIFs from previous runs — ~150 of them. Watch a few; they make the whole project click. |

---

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
  `final_areas.py`
- **token 2** → the satellite, lowercased and looked up in `landsat_bands.py` by
  `preprocess.get_common_bands()` (`helpers/preprocess.py:9`)

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

## Part 7 — The model contract (and the channel bug)

### Architecture

A U-Net with a ResNet-50 encoder, in PyTorch. Defined **three times**, in
`training/src/train.py`, `inference/cnn.py`, and inlined inside `inference/infer.py`.

```
input (N, 9, 128, 128)
   │
   ├─ ResNet-50 encoder, first conv replaced:
   │     nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
   │     ← ImageNet ResNet expects 3 channels; we have 9
   │  (ImageNet weights kept for all later layers; last two ResNet layers removed)
   │  → output: 2048 channels at 4×4
   │
   └─ decoder: 5 × ConvTranspose2d + BatchNorm + conv_block
         skip connections from encoder stages x2, x4, x5, x6
         plus the raw input concatenated at the last level
   │
   └─ classifier: Conv2d(16, 2, kernel_size=1) → 2-class logits
```

The two output channels are glacier and background, and one is the complement of the other. Two
channels rather than one only because PyTorch's cross-entropy expects `num_classes` outputs.

Glacier probability = `softmax(logits, dim=1)[:, 1]`, thresholded at 0.5.
Area = `mask.sum() * 0.0009` km² (900 m² per 30 m pixel), computed **after** resizing the prediction
back to the image's original dimensions.

### The checkpoints take 9 channels — and most of the code says 10

This is the single most important thing to understand in this repo, and the paper is what settles it.

The paper's channel figure lists exactly nine, with an editorial note reading **"Remove SWIR2 because
of quality"** and a caption reading "The 9 channels used in the CNN":

```
0: NDWI   1: NDSI   2: blue   3: green   4: red   5: nir   6: swir   7: thermal   8: DEM
```

I confirmed this against the checkpoint files themselves, without needing torch. A PyTorch `.pt` is a
zip of raw tensors, so the stem convolution's byte size gives its input channel count directly:

| Tensor | 9-channel size | 10-channel size | Found in both checkpoints |
|---|---|---|---|
| stem `Conv2d(C,64,7,7)` | **112,896 B** | 125,440 B | 112,896 ✅ |
| `c5` = `conv_block(32+C,16)` | **23,616 B** | 24,192 B | 23,616 ✅ |

Two independent tensors agree, in both `unet_summer_model_unfrozen_100` and `saved_models/model`.
**The trained models take 9 input channels.** `swir_2` was deliberately dropped for quality.

Now compare what each entry point actually builds:

| Entry point | `common_bands` | + DEM | + indices | Total | vs. a 9-ch checkpoint |
|---|---|---|---|---|---|
| `get_training_set.ipynb` | 6 — no `swir_2` | 7 | +NDSI +NDWI | **9** | ✅ **correct** |
| training data on disk | — | — | — | 7-channel tifs → **9** | ✅ **correct** |
| `final_areas.py:101` | 6 — no `swir_2` | 7 | +NDSI +NDWI **+ a duplicated NDWI** | **10** | ❌ shape error |
| `inference/infer.py:39` | 7 — *includes* `swir_2` | 8 | +NDSI +NDWI | **10** | ❌ shape error |
| `train.py` | hardcodes `Conv2d(10,…)`; normalize indexes `image[0]`…`image[7]` | — | — | 8→**10** | ❌ stale |

So the **data is right and the scripts are wrong.** The 7-channel training tifs on disk (verified:
`SamplesPerPixel: 7`) and the notebook that built them match the trained models exactly. Three
source files were never updated when `swir_2` was dropped.

The fixes are small:

- **`final_areas.py`** — delete the duplicated-NDWI line. Its own comment already tells you to:
  ```python
  inputs = torch.cat((ndwi, inputs), dim=1)   # REMOVE THIS LINE - only for testing
  ```
  Removing it takes 10 → 9 and makes the script correct. The comment is not a warning to ignore; it
  is the instruction.
- **`infer.py`** — drop `'swir_2'` from `common_bands` (8 → 7 → 9 total) and change the inlined
  `Conv2d(10, …)` to 9.
- **`train.py`** — change `Conv2d(10, …)` to 9, and fix the normalize block from 8 channels to 7.

**Why nobody noticed:** `final_areas.py` wraps every glacier in a bare `except:`, so a shape mismatch
prints `Error {glims_id}` and moves on. A full run over 245 glaciers can fail on *all* of them and
still exit cleanly with the three aggregate CSVs sitting there (empty). Always count the `Error` lines.

One subtlety: the hardcoded `10` in the class definitions does **not** prevent loading a checkpoint.
`torch.load` of a pickled module restores the real layer shapes and never re-runs `__init__`. The
hardcoded value only matters when you construct a fresh `UNet()` to train from scratch.

### Why the checkpoints are awkward to load

`torch.save(model)` saves the whole live Python object, not just the weights. Reading the pickle's
string table shows it references **`__main__.UNet`** and **`__main__.conv_block`** — because
`train.py` saved it while running as a script.

That means the class definitions must exist in the **`__main__`** namespace at load time. This
explains an otherwise baffling code smell: `infer.py` inlines a whole copy of the U-Net because it
*has to*, and `final_areas.py`'s `from inference.cnn import UNet, conv_block` works only because
`final_areas.py` is itself `__main__` when run as a script. Try loading a checkpoint from a fresh
notebook and you get:

```
AttributeError: Can't get attribute 'UNet' on <module '__main__'>
```

Consequences: editing `UNet`/`conv_block` can break old checkpoints, so change all three copies
together; `torch.load("model")` is cwd-relative in both scripts and no file named `model` exists; and
**converting to a `state_dict` is what makes these portable** — do that before sharing them anywhere.

### The two checkpoints

| File | Date | Size | Channels |
|---|---|---|---|
| `src/segmentation/unet_summer_model_unfrozen_100` | 2024-10-02 | 343,522,287 B | 9 |
| `src/segmentation/saved_models/model` | 2023-12-27 | 343,487,293 B | 9 |

Different SHA-256, ~10 months apart, and nothing on disk records which is which or how they differ.
`saved_models/` also holds **18 legacy Keras `.h5` files** (2022–2023, ~25 MB each) whose filenames
are a record of the band-selection experiments: `bl_gr_re_ni_sw_th_de_v{0,1,2}.h5` is
blue/green/red/nir/swir/thermal/dem, `re_ni_sw_de_v1.h5` is red/nir/swir/dem — and that last one is
the file `infer.py`'s commented-out Keras block used to load.

### The published training recipe

From the paper. Note it differs from the script's defaults in one important way:

| | Paper | `train.py` default |
|---|---|---|
| learning rate | 0.00001 | 0.00001 ✅ |
| **batch size** | **32** | **2** ❌ |
| encoder weights | unfrozen | unfrozen ✅ |
| loss | cross-entropy | `LOSS = 'ce'` ✅ |
| threshold | 0.5 | 0.5 ✅ |
| optimizer | Adam | Adam ✅ |
| data split | 6,456 train / 359 eval / 359 test, of 7,174 | 90/10, no eval set |
| model selection | epoch minimizing eval cross-entropy | whatever the last epoch gives |

Hyperparameters were tuned on a random 4,000-image subset (3,600 train / 400 eval, 20 epochs):

| LR | Batch | Cross-entropy |
|---|---|---|
| 0.001 | 16 | 3.35 |
| 0.0001 | 16 | 0.68 |
| 0.00001 | 16 | 0.50 |
| 0.00001 | 8 | 0.53 |
| **0.00001** | **32** | **0.49** ← chosen |

Augmentation: vertical flip, horizontal flip, and a 3×3 Gaussian blur (σ = 0.8), each applied
**independently with probability 0.5**. That's a ~75% increase in training data (to 11,298 images),
not the 4× you'd get from applying all three to every image — so p(no transform) = 0.125.

Reported result: **Dice 0.92** on the test set. The ablation trail, from the paper's notes:

| Change | Dice |
|---|---|
| 3,000 images, no transfer learning | 0.83 |
| + transfer learning (ImageNet ResNet-50) | 0.86 |
| + NDSI & NDWI channels | 0.88 |
| + full dataset, cross-entropy loss | **0.92** |
| (same, Dice loss instead) | 0.90 |
| frozen encoder | 0.91 |
| ResNet-18 instead of ResNet-50 | 1–2% worse |

Threshold sweep: 0.1 → 0.16, 0.4 → 0.923, **0.5 → 0.926**, 0.8 → 0.87. So 0.5 is empirically
optimal, not just conventional.

Cross-entropy was chosen over Dice loss deliberately: it penalizes *confidence*, not just
correctness, which suits blurry ice/rock boundaries. Class imbalance is mild here (glacier and
background pixels are roughly balanced in these crops), so focal loss wasn't needed.


## Part 8 — What's missing, broken, or unbacked

Established by diffing local inventories against S3 and by reading the code.

### 🔴 Blocking — you cannot run inference without fixing these

| # | Problem | Detail |
|---|---|---|
| 1 | **All inference imagery is absent locally** | `full_time_series_c02_t1_l2/landsat/` is empty. It used to live on an external SSD — `time_series_metadata_handler.ipynb` still points at `/Volumes/T7/GlacierView/ee_landing_zone/...`, which isn't mounted. Good news: nothing is missing *upstream*. All 245 glaciers in `filtered_inference_data.csv` and all 1,024 in `geog_area_rollup_250.csv` exist in S3. |
| 2 | **No file named `model`** | Both inference scripts `torch.load("model")` relative to cwd. Copy or symlink `src/segmentation/unet_summer_model_unfrozen_100`. |
| 3 | **`data_label` points at an empty directory** | `infer.py:31` sets `data_label = "full_time_series"`, and `final_areas.py:90`/`:94` inline that same string directly into the path — no `_c02_t1_l2` suffix in either. That directory exists but is empty; all real data is under `full_time_series_c02_t1_l2`. Fix the constant or symlink. |
| 4 | **`glacier_areas/` is never created** | `final_areas.py` writes `f"glacier_areas\\{glims_id}_areas.csv"` — a directory it never `mkdir`s, using a **Windows path separator on macOS**. Every iteration throws into the bare `except`, so the per-glacier CSVs silently never appear while the three aggregate CSVs do. A run looks successful. |
| 5 | **Both inference scripts build 10 input channels; the checkpoints take 9** | Every glacier throws a shape error into the bare `except`, so a full run produces nothing but `Error` lines and three empty aggregate CSVs. A three-line fix — see Part 7. |
| 6 | **`train.py` is stale against both the data and the checkpoints** | Its normalize block indexes `image[7]` on 7-channel tifs (`IndexError`), and it hardcodes a 10-channel stem. The data is correct; the script is not. See Part 7. |

### 🟡 Missing but regenerable

`areas_no_threshold.csv`, `areas_05_thresh.csv`, `areas_binary_05.csv` — `final_areas.py` creates
these on startup.

### ✅ Recovered — the four analysis CSVs (2026-09-19)

These were missing when this document was first written (the LLD recorded them as *"somansh will push
to github and email me"*). They have since been added and live in **`data/analysis/`**, committed —
the blanket `*.csv` ignore carries a negation for that directory. The notebook resolves them through
an `ANALYSIS_DIR` constant derived from its own location, so it works wherever Jupyter starts.

| File | Shape | What it actually is |
|---|---|---|
| `geo_areas.csv` | 18,093 × 32 | **The `glims_18k` table itself.** Full GLIMS attributes per glacier: `glac_id`, `db_area`, `area`, `geog_area`, **`geog_area_rollup`**, `bboxes`, `min/mean/max_elev`, `src_date`, `glac_name`, `analysts`, `submitters`. This is the join key for regional grouping. |
| `training_data_set.csv` | 10,447 × 12 | Output of `training_data_query.sql` — one row per candidate training image with `cloud_cover`, `image_quality`, `num_pixels`, `percentage_zero_pixels`, `rank_score`. This is the paper's stale "10,443"; the empty-mask variance filter reduces it to the 7,174 actually trained on. |
| `training_areas.csv` | 3 × 260 | A small transposed lookup — ~259 glacier IDs mapped to areas. |
| `areas_binary_05_filtered_smooth_25.csv` | 541 dates × **245** glaciers | Smoothed `final_areas.py` output in wide format. The 541 rows are monthly timestamps — exactly `pd.date_range('1979-01-01','2024-01-01',freq='MS')`, which is the range hardcoded in `final_areas.py`. |

Two things to notice:

⚠️ **The areas CSV covers 245 glaciers, but the paper reports 1,083.** So this is the output of the
older 50-glaciers-per-region run, *not* the run behind the paper's headline −0.199%/yr. Whatever
produced the paper's numbers is still unaccounted for. If you're trying to reproduce the paper, this
is your first blocker.

⚠️ **`geo_areas.csv` includes an Oceania rollup (35 glaciers)** — Asia 12,625 / North America 3,704 /
South America 872 / Europe 602 / Caucausus 255 / Oceania 35. The SQL treats Oceania as southern
hemisphere, but it never appears among the paper's five regions. Those 35 glaciers drop out somewhere
between selection and results, and nothing documents where.

### 🔵 Present locally but backed up nowhere

This is the risk that should worry you most, because it's silent:

| Asset | Situation |
|---|---|
| ~~Model checkpoints~~ | ✅ **Resolved 2026-09-19.** Both PyTorch checkpoints and all 18 legacy Keras files are now in `s3://segmentation-model-gv/checkpoints/` (us-east-1), alongside a `MANIFEST.md` recording channel counts, SHA-256s and the training recipe. 20 objects, byte-verified. ⚠️ Bucket **versioning is still off** — an overwrite cannot be undone; enable it before re-uploading. |
| 148 training glaciers | Local has 18,093 glacier dirs; `raw-training-images-t1-l2-sr` has 17,945. 148 exist only locally. |
| All 18,093 training-landing-zone DEMs | There is **no training DEM bucket at all**. |
| Derived metadata CSVs | `denormalized_metadata.csv`, `filtered_training_data.csv`, `filtered_training_data_summer_months.csv`, `filtered_inference_data.csv`, `net_new_glims_ids_for_segmentation.csv` — the buckets hold only the four crawled tables, not these. |
| `src/glims/data/glims_db_20210914/` | The source GLIMS inventory, 20 files. Only the *derived* `glims_18k.parquet` is in S3. (Re-downloadable from GLIMS, but it's a dated snapshot — a fresh download would differ.) |
| `src/earth_engine/` source code | The entire directory is untracked in git. The `EePull` class exists only on this machine. |

### ✅ Confirmed complete and healthy

- **Ready-to-train data** — `processed_training_data_summer_months/` holds 7,174 images + 8,902
  masks, an exact mirror of `training-images-t1-l2-sr` (7,174 + 8,902 + 2 `.DS_Store` = the bucket's
  16,078 objects). (Caveat: it's the 7-channel vintage — see Part 7.)
- **Raw training data** — 18,093 glacier dirs of imagery and DEMs, plus `masks_staging_2/`.
- **Inference DEMs** — all 1,122, byte-identical ID set to the bucket.
- **Inference metadata** — the four table CSVs plus `filtered_inference_data.csv`.

One curiosity: **21 glaciers have DEMs in S3 but no imagery.** All are far-north (Ellesmere Island),
consistent with the SQL comment about excluding Canada for lack of DEM coverage — so they look
deliberately dropped after the DEM pull, not lost.

---

## Part 9 — Getting set up

### Step 0 — install the environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. One command:

```bash
uv sync
```

That reads `pyproject.toml`, installs the exact versions pinned in `uv.lock` into `.venv/`, and
downloads CPython 3.12 if you don't already have it (the version is pinned in `.python-version`).
Then either prefix commands with `uv run`, or activate `.venv` the usual way.

```bash
uv run python src/segmentation/inference/infer.py --glimsid G007026E45991N
uv run jupyter lab          # notebook deps are in the "notebook" group, installed by default
```

Verified working: all 24 third-party imports resolve on Python 3.12, and a released checkpoint loads
and runs a forward pass. Dependencies were derived by scanning every `import` in `src/` rather than
inherited from the old `requirements.txt`.

**Why `requires-python` is capped at `<3.13`:** the geospatial stack (rasterio, geopandas) and torch
lag behind new interpreter releases. Left uncapped, uv resolves against 3.14 and you get wheels that
don't exist. If you widen it, re-run `uv lock` and actually `uv sync` to check.

**If you need a specific CUDA build:** PyPI's torch gives a CPU/MPS build on macOS and a
bundled-CUDA build on Linux, which covers both a laptop and a GPU box. The published model was
trained against cu118 — `pyproject.toml` carries a commented `[[tool.uv.index]]` block for pinning
that exactly.

*Historical note, in case you find old instructions:* `requirements.txt` was removed in favour of
`pyproject.toml`. It could not be installed as written — it was UTF-16 encoded with CRLF line
endings, pinned `torch~=2.1.0+cu118` (no such wheel exists for macOS), and omitted 13 packages the
code imports. The `src/segmentation/training/src/requirements.txt` beside it was a 200-line frozen
`pip freeze` of someone's entire 2021 environment, including `pywin32`.


### Step 2 — credentials

```bash
aws sts get-caller-identity     # confirm you are in the project AWS account
```

For Earth Engine (only needed if you're pulling *new* imagery — you probably aren't yet):
`ee.Authenticate()` once in a notebook, then `ee.Initialize()`.

### Step 3 — fix the hardcoded paths

Notebooks hardcode `~/Desktop/projects/GlacierView` as the project root and `sys.path.insert` the
helpers directory. If your checkout is elsewhere, you'll be editing that line in every notebook.
Scripts under `src/segmentation/` use `Path(__file__).parent...` and are fine.

### Step 4 — smoke test on one glacier

Don't start with 486 GB. Start with Trient (`G007026E45991N`), ~700 MB:

```bash
# 1. get one glacier's imagery
aws s3 cp s3://full-time-series-images-t1-l2-sr/G007026E45991N/ \
  src/earth_engine/data/ee_landing_zone/full_time_series_c02_t1_l2/landsat/G007026E45991N/ \
  --recursive --region us-west-1 --exclude "*/meta_data/*" --exclude "*.DS_Store"

# 2. its DEM should already be local; confirm
ls src/earth_engine/data/ee_landing_zone/full_time_series_c02_t1_l2/dems/G007026E45991N_NASADEM.tif

# 3. put the checkpoint where the script looks
cd src/segmentation/inference
ln -s ../unet_summer_model_unfrozen_100 model

# 4. fix data_label in infer.py: "full_time_series" → "full_time_series_c02_t1_l2"

# 5. run
python infer.py --glimsid G007026E45991N
```

Success looks like a GIF in `src/segmentation/gifs/G007026E45991N.gif` showing the ice boundary over
time. Compare it against the ~150 GIFs already in that directory from previous runs.

**Trient is also your regression test**, because the paper documents it in detail: **464 images** from
**1984-04-18 to 2023-10-07**, Landsat 5 and 7. Its area declines overall, with most of the loss
concentrated in **1985–1988** and **2002–2015**, and roughly stable over the last decade; the ice is
lost at the northern terminus. If your run reproduces that shape, the pipeline is working.

Two caveats on that comparison. The paper's Trient run used *looser* filters than the main SQL —
cloud cover < 20, zero pixels < 15%, quality ≥ 9, no-data = 0, pixels > 50,000 — so image counts
won't match unless you match the filters. And a commented-out passage notes the Swiss GLAMOS
inventory measured Trient at 5.76 km² in 2006 while the model predicted 4.95 km², a **14%
underestimate** — so expect a level offset even when the trend is right.

Exclude `.DS_Store` (some are 660 KB — they got uploaded with the tifs) and `meta_data/` (only needed
for re-running the metadata handler; `read.get_rasters` only globs `.tif` anyway).

### Step 5 — scaling up, without pulling 486 GB

The bucket is 486 GB, but the filter narrows inference to **245 glaciers / 14,952 files / 23.7 GB**
(computed by summing `file size in bytes` from `file_attributes_250.csv` for every file in
`filtered_inference_data.csv` — all 14,952 matched).

`final_areas.py` calls `read.get_rasters(glacier_dir)` **without** the `filtered_file_path`
argument, so it processes whatever is on disk. **Downloading only the filtered files is therefore how
the quality filter gets applied at inference time.** Pull that subset, not the bucket.

(`read.get_rasters` *does* accept `filtered_file_path` — `helpers/read.py:37` — it's just not used
by `final_areas.py`. Passing it would be a cleaner fix than relying on what you downloaded.)

The LLD notes `aws s3 sync` was too slow for the original upload and `cp --recursive` was used.

### Step 6 — training, if you go there

```bash
cd src/segmentation/training/src
ln -s ../data/processed_training_data_summer_months training_data
python train.py --epochs 10 --lr 0.00001 --decay 0.00001 --batch 2
```

This **will fail** on the channel mismatch (Part 7) until that's resolved: the normalize block
indexes `image[7]` on 7-channel data, and the stem is hardcoded to 10 channels. `train.py` also
expects hyperparameters you can't set from the CLI — `LOSS`, `TRANSFORMS`, `NORMALIZE`,
`UNFREEZE_WEIGHTS` are module-level constants you edit in the file.

To match the published recipe use `--batch 32` (not the default 2), and be aware the script has no
eval split and keeps the last epoch rather than the best one. See the recipe table in Part 7.

### What doesn't exist

No test suite, no CI, no Docker, no orchestration. Dependencies are managed (`pyproject.toml` +
`uv.lock`) and linting is set up — `uv run ruff check .` passes repo-wide. The rule set is pinned
explicitly in `[tool.ruff.lint]` rather than inherited from ruff's defaults, which widen between
releases. Notebooks carry documented per-file exemptions for rules that fire on normal notebook
idioms. `ruff format` is *not* enforced: it would reformat 31 files, mostly exploratory notebooks,
for no maintainability gain.
`git status` on a fresh clone will show notebook diffs immediately, because notebooks store their
output cells — consider `nbstripout` if that bothers you.

---

## Part 10 — Landmines, ranked

Ordered by how likely they are to waste your day.

1. **The silent `except:` in `final_areas.py`.** Every glacier is wrapped in
   `try: … except: print(f"Error {glims_id}")`. A run over 245 glaciers can fail on 244 of them and
   still exit 0 with output files present. **Count the `Error` lines before trusting any output.**
2. **The 9-vs-10 channel mismatch.** Both inference scripts and `train.py` build 10 input channels;
   both checkpoints take 9. `final_areas.py` errors on every glacier (silently, see #1);
   `infer.py` fails the same way. The data on disk is correct — the scripts are stale. Three-line
   fix in Part 7.
3. **`full_time_series`** (no `_c02_t1_l2`) is the dataset name baked into both inference scripts —
   `infer.py:31` as `data_label`, `final_areas.py:90`/`:94` inlined into the path — and it points at an
   empty legacy directory. You'll get "no such file" or an empty dict and an unhelpful downstream error.
4. **Three copies of the U-Net.** Edit one and the other two drift; checkpoints are pickled modules,
   so drift breaks loading.
5. **`torch.load("model")` is cwd-relative** in both scripts, and no `model` file exists.
6. **`num_pixels` and `percentage_zero_pixels` are band-inflated** (`h × w × num_of_bands`). Don't
   reason about them as ground-pixel counts. The LLD calls this out as a known mistake.
7. **The SQL cloud threshold is `< 5`; the LLD says 10; Earth Engine downloaded at `<= 10`.** Decide
   which you mean before re-running any filter.
8. **`infer.py` plots `sqrt(sum(prediction))`** and labels it "Estimated Surface Area (no units)".
   That is not an area. Use `final_areas.py` for real numbers.
8b. **`train.py --batch` defaults to 2; the paper used 32.** Don't reproduce training with the
   defaults and expect the published Dice score.
9. **Two dead Keras files** (`helpers/model.py`, `inference/model.py`) sit next to the live PyTorch
   code and import TensorFlow. Don't extend them; don't install TF for them.
10. **A stray `full_time_series_c02_t1_l2/G007026E45991N/`** sits *beside* `landsat/` instead of
    inside it — an old Trient download that landed a level too high. Harmless but confusing.
11. **`cnn.py` defines `conv_block` twice** (lines 7 and 207). Verified byte-identical, so the
    shadowing is harmless — but if you edit one, edit both.
12. **`SHUFFLE_DATASET = False`** in `train.py`. Unusual; be deliberate about it.
13. **`.DS_Store` files are in S3**, some 660 KB, mixed in with the tifs. Always `--exclude` them.
14. **Notebooks hardcode one person's home directory** and some point at an unmounted external SSD.
15. **Region flags.** Forget `--region` and `aws s3` will fail confusingly on a cross-region bucket.

---

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
   design — `infer.py`, `final_areas.py` and `train.py` are all stale on channel count.
5. **`low_level_design.md`** — authoritative on *rationale* (why thresholds, what the tech debt is,
   what the risks were). Its *specifics* have drifted: Glue table names are mangled, the band order
   is listed NDSI-first when the code prepends NDWI last, and its cloud-cover threshold disagrees
   with the SQL.
6. **`README.md`** — the oldest of these (Nov 2023). Its directory layout
   (`src/segmentation/training/src`, `inference/src`) partly predates the current tree, and it
   describes the model as taking "128x128 pixels with 7 bands" — which is neither the 8-channel
   design nor the 10-channel model input. Treat as historical.

### Open questions

**Answered by the paper** (previously open in this doc):

- ~~Which channels was the model trained on?~~ **Nine**: NDWI, NDSI, blue, green, red, nir, swir,
  thermal, DEM. `swir_2` was dropped for quality. Confirmed directly against both checkpoints.
- ~~Where are the four analysis CSVs?~~ Recovered 2026-09-19, now in the repo root.

**Still open** — these need Somansh Budhwar or Armin Schwartzman (`armins@ucsd.edu`):

1. **Where is the 1,083-glacier areas output?** The paper's headline −0.199%/yr comes from 1,083
   glaciers / 505,835 images, but the recovered `areas_binary_05_filtered_smooth_25.csv` covers only
   245. The artifact behind the published numbers is missing.
2. **Does the 250-glacier selection SQL exist anywhere?** The committed file is 0 bytes, yet 1,101
   glaciers were downloaded and the `_250` Athena tables are populated. The query that chose them is
   unrecorded.
3. **Cloud-cover threshold: 5 or 10?** The committed SQL says `< 5`, the LLD and the Earth Engine
   download filter say 10, and the paper's Trient case study says `< 20`. Three different numbers.
4. **Is the observation weighting in the regression correct?** The paper's formula computes the
   proportion of data *missing* while the text claims it upweights *complete* series. One of the two
   is wrong, and it affects every published regional average.
5. **What distinguishes the two PyTorch checkpoints?** `saved_models/model` (Dec 2023) vs
   `unet_summer_model_unfrozen_100` (Oct 2024). Same architecture, different weights, no record of
   which produced the paper's 0.92 Dice.
6. **Where did the 35 Oceania glaciers go?** Present in `geo_areas.csv`, treated as southern
   hemisphere by the SQL, absent from the paper's five regions.
7. **Is the GLAMOS comparison still current?** A commented-out passage notes Trient measured
   5.76 km² in 2006 against the model's 4.95 km² — a 14% underestimate. If that holds, it belongs in
   the paper's limitations.
