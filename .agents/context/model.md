# Model contract

The published model is described in the project's IEEE paper
(`.agents/references/paper.md`). It is defined **once**, in `glacierview/models/unet.py`;
`infer.py`, `glacierview areas` and `glacierview train` all import `UNet` and `conv_block` from
there. It used to be copied into all three, which is how they drifted apart.

### The checkpoints take 9 input channels, not 10

Verified by inspecting the serialized tensors in both checkpoints (stem conv is
`Conv2d(9,64,7,7)` = 112,896 bytes; `c5` is `conv_block(41,16)` = 23,616 bytes — the 10-channel
variants would be 125,440 and 24,192 and appear in neither file):

```
0: NDWI   1: NDSI   2: blue   3: green   4: red   5: nir   6: swir   7: thermal   8: DEM
```

`swir_2` was **deliberately dropped for quality reasons** — the paper's channel figure says so
explicitly ("Remove SWIR2 because of quality"), and its caption reads "The 9 channels used in the
CNN." Three source files originally built 10 channels against these 9-channel checkpoints, so every
glacier failed; that is fixed, and the count is now derived from one constant:

| Source | Channels | |
|---|---|---|
| `glacierview.preprocess` `COMMON_BANDS` | 6 bands + DEM = 7 → +NDSI +NDWI = **9** | matches |
| training data on disk | 7-channel tifs (verified: `SamplesPerPixel: 7`) → **9** | matches |
| `glacierview.models.unet` `IN_CHANNELS` | **9** | the single declaration |

`infer.py`, `glacierview areas` and `glacierview train` each assert their built tensor equals `IN_CHANNELS`
before it reaches the first convolution, so a future mismatch fails loudly rather than inside a
bare `except`.

### Checkpoints are pickled modules referencing `__main__`

`torch.save(model)` saved the whole live object from a script, so the pickle references
`__main__.UNet` and `__main__.conv_block` (verified by reading the pickle's string table). To load a
checkpoint, those two names must exist in the **`__main__`** namespace — which is why `infer.py`
inlines a copy of the class, and why `glacierview areas`'s `from inference.cnn import UNet, conv_block`
works only because it is itself `__main__` when run as a script. Loading from a notebook or library
needs the same names injected first. Consequences:

- Edit `UNet`/`conv_block` and old checkpoints may stop loading.
- `torch.load("model")` is **cwd-relative** in both scripts and no file named `model` exists — copy
  or symlink a checkpoint first.
- Converting to a `state_dict` is what makes these portable; do that before sharing them anywhere.

### Other invariants

- ResNet-50 stem replaced with a 9-channel `Conv2d`; ImageNet weights loaded for the rest; encoder
  output is 2048 channels at 4×4. Last two ResNet layers removed.
- Output: 2-class logits (glacier / background, complementary). Probability is
  `softmax(...)[:, 1]`, thresholded at 0.5.
- Area = `mask.sum() * 0.0009` km² (30 m × 30 m pixels), computed **after** resizing the prediction
  back to the image's original dimensions.
- NDSI/NDWI are computed from the raw band stack at fixed indices (`green=ch1`, `nir=ch3`,
  `swir=ch4`) then *prepended*, NDSI first and NDWI second — so NDWI lands at index 0.

### Two checkpoints exist

| File | Date | Size | Notes |
|---|---|---|---|
| `src/segmentation/unet_summer_model_unfrozen_100` | 2024-10-02 | 343,522,287 B | the production checkpoint; 9-channel |
| `src/segmentation/saved_models/model` | 2023-12-27 | 343,487,293 B | earlier, distinct SHA-256; also 9-channel |

`saved_models/` also holds 18 legacy Keras `.h5` files (2022–2023) whose names record the band
experiments (`bl_gr_re_ni_sw_th_de_v{0,1,2}.h5`, `re_ni_sw_de_v1.h5`, …). All of it is backed up to
`s3://segmentation-model-gv/checkpoints/`, with a `MANIFEST.md` recording channel counts, SHA-256s
and the training recipe. Bucket versioning there is still disabled, so an overwrite is unrecoverable.

### Published training recipe (from the paper — differs from the script defaults)

| | Paper | `glacierview train` default |
|---|---|---|
| learning rate | 0.00001 | 0.00001 ✅ |
| **batch size** | **32** | **2** ❌ |
| encoder | unfrozen | `UNET(freeze_encoder=False)` ✅ |
| loss | cross-entropy | `LOSS = 'ce'` ✅ |
| threshold | 0.5 | 0.5 ✅ |
| optimizer | Adam | Adam ✅ |
| splits | 6,456 train / 359 eval / 359 test of 7,174 | 90/10 only, no eval set |
| model selection | min cross-entropy on the eval set | last epoch |

Augmentation: vertical flip, horizontal flip, and 3×3 Gaussian blur (σ = 0.8), each applied
independently with p = 0.5 — growing the training set ~75%, to 11,298 images. Reported test
**Dice = 0.92**.

Ablations the paper records: 3,000 images without transfer learning 0.83 → with transfer learning
0.86 → adding NDSI/NDWI 0.88 → full dataset with cross-entropy **0.92** (Dice loss gave 0.90).
Frozen encoder 0.91 vs unfrozen 0.92. ResNet-50 beat ResNet-18 by 1–2%. Threshold sweep:
0.1 → 0.16, 0.4 → 0.923, **0.5 → 0.926**, 0.8 → 0.87.
