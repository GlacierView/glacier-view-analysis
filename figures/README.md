# Figures

Diagrams and plots for the paper and the docs. Several filenames are opaque, so
here is what each one is.

## Pipeline and architecture

| File | What it shows |
|---|---|
| `cnn_architecture.png` | The U-Net: ResNet-50 encoder, decoder, skip connections. Used in `README.md`. |
| `data_collection.png` | The training and inference data-collection workflow, end to end. |
| `data_collection_and_preprocessing.drawio` | **Editable source** for the workflow diagram — open at [app.diagrams.net](https://app.diagrams.net). Edit this rather than the exported PNG. |
| `data_collection_and_preprocessing-Data collection.drawio.png` | Export of the above. |
| `training_data_processing.png` | How raw scenes and GLIMS polygons become image/mask training pairs. |

## Data distributions

| File | What it shows |
|---|---|
| `distribution_of_glacier_surface_area.jpg` | Glacier area across GLIMS, with the 1 km² line marked. Motivates the 1–50 km² selection filter. |
| `distribution_of_metadata.jpg` | Landsat metadata distributions under the quality filters. This is where the cloud-cover and zero-pixel thresholds came from. |
| `glims_subsets.png` | How the GLIMS snapshot narrows to the 18,093 selected glaciers. |

## The UTM problem

The same glacier can be delivered in different UTM zones on different dates,
which is why masks are reprojected per image rather than once globally. See the
"Files, projections, and why the same glacier moves" section of
`docs/ONBOARDING.md`.

| File | What it shows |
|---|---|
| `utm zone 31 trient landsat 7.png` | Trient in UTM zone 31 (Landsat 7). |
| `utm zone 32 trient landsat 5.png` | The same glacier in zone 32 (Landsat 5). |
| `utm zone 31-32 overlap trient landsat 7-5.png` | The two overlaid — the offset is the problem. |

The `.pgw` files are world files: the georeferencing for the matching `.png`.

## `inventory-examples/`

Screenshots of glacier outlines from two GLIMS contributors, kept as the
evidence behind excluding them from the training set:

- `rgi-1..6.png` — Randolph Glacier Inventory. Prioritised complete global
  coverage over per-outline accuracy.
- `globglacier-1..6.png` — GlobGlacier (ESA). A smaller, more carefully
  curated set.

## Logos

`earth-engine-logo.png`, `glims_logo_smooth.png` — for slides and diagrams.
