# Model checkpoint manifest

Generated 2026-09-19. Records what each checkpoint is, because nothing else on disk does.

**None of these files is in git** (`.gitignore` excludes `src/segmentation/saved_models/`)
**and none is in S3.** They exist only on this machine. See `docs/ONBOARDING.md` Part 8.

## PyTorch checkpoints

Both take **9 input channels**, verified from the serialized tensor sizes (stem conv
`Conv2d(9,64,7,7)` = 112,896 B; `c5` = `conv_block(41,16)` = 23,616 B). Channel order:

```
0: NDWI  1: NDSI  2: blue  3: green  4: red  5: nir  6: swir  7: thermal  8: DEM
```

`swir_2` was deliberately excluded for quality reasons (per the paper's channel figure).

Saved with `torch.save(model)` — a **pickled module referencing `__main__.UNet` and**
**`__main__.conv_block`**, not a state dict. Both names must exist in `__main__` before
`torch.load`. Convert to a state dict before sharing these anywhere.

### `unet_summer_model_unfrozen_100`

- modified: 2024-10-02
- size: 343,522,287 bytes
- sha256: `9c29c41515bde39c1722c5bd18f5e82711fb54c86cd5f53d5262d9348623c2b8`
- input channels: 9
- Production checkpoint. The paper's reported model (Dice 0.92) is believed to be this one — UNCONFIRMED.

### `saved_models/model`

- modified: 2023-12-27
- size: 343,487,293 bytes
- sha256: `da5296e839fa4e7e6fc601735273757b329531abee01839abcf4b8f6c8f60237`
- input channels: 9
- Earlier PyTorch checkpoint. Distinct weights from the above. Relationship to the paper unknown.

### Published training recipe (from the paper)

| Parameter | Value |
|---|---|
| architecture | U-Net, ResNet-50 encoder (ImageNet weights), last two ResNet layers removed |
| input | (N, 9, 128, 128), values normalized to [0,1] |
| output | 2-class logits; glacier probability = softmax(...)[:,1], threshold 0.5 |
| training set | 7,174 images (one temporal composite per glacier) → 6,456 train / 359 eval / 359 test |
| augmentation | vflip, hflip, 3x3 Gaussian blur (sigma 0.8), each independently at p=0.5 → 11,298 images |
| learning rate | 0.00001 |
| batch size | 32 |
| optimizer | Adam |
| loss | cross-entropy |
| encoder | unfrozen (fine-tuned) |
| model selection | epoch minimizing eval-set cross-entropy |
| reported test Dice | 0.92 |

Area conversion: `mask.sum() * 0.0009` km² (30 m x 30 m pixels), after resizing the
prediction back to the source image's original dimensions.

## Legacy Keras checkpoints (`saved_models/*.h5`)

The TensorFlow era, 2022–2023, superseded by the PyTorch model. Retained as the record of
the band-selection experiments — the filenames encode which bands each used
(`bl`=blue, `gr`=green, `re`=red, `ni`=nir, `sw`=swir, `th`=thermal, `de`=DEM).
`re_ni_sw_de_v1.h5` is the file `glacierview infer`'s commented-out Keras block loaded.
The Keras code that built these (`helpers/model.py`, `inference/model.py`) is dead.

| File | Modified | Bytes | sha256 |
|---|---|---|---|
| `8_16_21-500t-6b-a.h5` | 2022-10-03 | 25,709,336 | `04e053ede418c563…` |
| `8_25_21-1136t-7b-a.h5` | 2022-10-03 | 25,711,064 | `2a89154377b35269…` |
| `8_25_21-1908t-7b-a.h5` | 2022-10-03 | 25,711,064 | `b1b75ccdf1d45c92…` |
| `bl_gr_re_ni_sw_th_de_v0.h5` | 2022-11-21 | 25,711,080 | `7af82771b0ce74e7…` |
| `bl_gr_re_ni_sw_th_de_v1.h5` | 2022-11-21 | 25,711,080 | `c611b557dbf60940…` |
| `bl_gr_re_ni_sw_th_de_v2.h5` | 2023-01-22 | 25,711,080 | `2fc6dfc77299c860…` |
| `four_band_model.h5` | 2022-10-03 | 25,705,784 | `535908751301b92a…` |
| `largest_ee.h5` | 2022-10-03 | 25,709,336 | `e6ce600fbcb12ea6…` |
| `largest_model.h5` | 2022-10-03 | 25,705,784 | `0f7596aeb6d96f34…` |
| `model_large.h5` | 2022-10-03 | 25,705,784 | `8f78063469600468…` |
| `no_23.h5` | 2022-10-03 | 25,705,784 | `be03428b4417b12f…` |
| `no_post_trient.h5` | 2022-10-03 | 25,705,784 | `6896d69fb00ff71e…` |
| `no_trient.h5` | 2022-10-03 | 25,705,784 | `9e1a4d05b9ffdf30…` |
| `re_ni_sw_de_v1.h5` | 2023-02-13 | 25,705,800 | `43fb5b70bb88168a…` |
| `test.h5` | 2022-10-23 | 25,709,352 | `ec6c68eaacc61bf3…` |
| `unaugmented_model.h5` | 2022-10-03 | 25,706,864 | `418e4ad1152b9ef7…` |
| `unaugmented_model_fixed_masks.h5` | 2022-10-03 | 25,705,784 | `10fc51f3e706eeaa…` |
| `with_trient.h5` | 2022-10-03 | 25,705,784 | `eb9c17da2b27afd0…` |

Full hashes: run `shasum -a 256 saved_models/*.h5`.

## Measured scores

Both checkpoints scored on the same 359-image slice (`build_manifest` with
`seed=0`, the second half of the 10% holdout), threshold 0.5:

| Checkpoint | Dice | Jaccard |
|---|---|---|
| `unet_summer_model_unfrozen_100` | **0.8681** | 0.7679 |
| `unet_2023-12-27` | 0.8261 | 0.7055 |

`unet_summer_model_unfrozen_100` is the published model — confirmed by the
project lead and consistent with these scores and with its filename, which
encodes the recipe (summer images, unfrozen encoder, 100 epochs).

⚠️ **Neither reaches the paper's reported 0.92, and batching does not explain
it.** Averaging Dice different ways over the same predictions:

| Averaging | Dice |
|---|---|
| per image | 0.8612 |
| batches of 8 | 0.8681 |
| batches of 32 (the paper's batch size) | 0.8713 |
| pooled over all pixels | 0.8692 |

All land at 0.86–0.87, so the ~0.05 gap is real rather than a measurement
convention. Note the gap should if anything be *smaller* than reported here:
these 359 images were almost certainly in the checkpoint's training data,
since the original script shuffled unseeded with no persistent split, so this
slice has no relationship to whatever it held out. That biases the score up.

Three candidate explanations, not distinguished:

1. The 0.92 was measured on a different test split, which was never recorded.
   Dice varies considerably with which glaciers land in the holdout.
2. The 0.92 came from a third checkpoint that no longer exists.
3. **The 0.92 is a different quantity.** The original script used
   `torchmetrics.Dice(average='micro')` on class indices, which counts
   background agreement as well as ice overlap and scores much higher than the
   mask-overlap Dice computed here. This is the leading suspicion, and it
   would mean the published figure is not comparable to what the current code
   reports.

## Remaining unknowns

1. What changed between the 2023-12-27 and 2024-10-02 checkpoints.
2. Neither has recorded per-region metrics, training duration, or the exact training-set snapshot.

Ask Somansh Budhwar or Armin Schwartzman (armins@ucsd.edu).
