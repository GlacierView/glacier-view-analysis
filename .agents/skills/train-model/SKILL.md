# Train the model

## When to use

To retrain or fine-tune the segmentation U-Net.

## Workflow

```bash
cd src/segmentation/training/src
ln -s ../data/processed_training_data_summer_months training_data
uv run python train.py --epochs 10 --lr 0.00001 --decay 0.00001 --batch 32
```

`train.py` expects `training_data/{images,masks}` relative to its working directory. It writes each
run to `experiments/<counter>/` (hyperparameters, loss plot, dice histogram, `model`) and also
overwrites `src/segmentation/inference/model`.

## Match the published recipe

The script's defaults are not what the paper used:

| | Paper | `train.py` default |
|---|---|---|
| learning rate | 0.00001 | 0.00001 |
| **batch size** | **32** | **2** |
| encoder | unfrozen | unfrozen |
| loss | cross-entropy | `LOSS = 'ce'` |
| splits | 6,456 train / 359 eval / 359 test | 90/10, no eval set |
| model selection | min eval cross-entropy | last epoch |

`LOSS`, `TRANSFORMS`, `NORMALIZE` and `UNFREEZE_WEIGHTS` are module-level constants you edit in the
file, not flags. `freeze_encoder=False` starts the ResNet trainable, which is the published setting;
`UNFREEZE_WEIGHTS` is a separate knob that unfreezes mid-training and only matters if you start
frozen.

## Notes

- Training data on disk is 7-channel; NDSI and NDWI bring it to the 9 the model takes. `train.py`
  asserts this per image.
- Reported test Dice is 0.92. Full ablation trail in `.agents/references/paper.md`.

## References

- `.agents/context/model.md` — architecture and the checkpoint contract
- `src/segmentation/MODEL_MANIFEST.md` — what each existing checkpoint is
