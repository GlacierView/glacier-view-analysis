# Train the model

## When to use

To retrain or fine-tune the segmentation U-Net.

## Workflow

```bash
cd src/segmentation/training/src
ln -s ../data/processed_training_data_summer_months training_data
uv run python train.py --epochs 10 --lr 0.00001 --decay 0.00001 --batch 32
```

`glacierview train` expects `training_data/{images,masks}` relative to its working directory. It writes each
run to `experiments/<counter>/` (hyperparameters, loss plot, dice histogram, `model`) and also
overwrites `src/segmentation/inference/model`.

## Match the published recipe

The script's defaults are not what the paper used:

The CLI defaults now *are* the published recipe — learning rate 1e-5, batch
size 32, unfrozen encoder, cross-entropy, a 90/5/5 split, and the epoch that
minimises eval loss. The old script defaulted to batch size 2, had no eval
split, and kept whatever the last epoch produced.

Two bugs in the old loop are fixed, and both change training behaviour:
`optimizer.zero_grad()` was never called, so gradients accumulated across a
whole epoch; and only the last batch's loss was recorded. Reproducing the
paper's exact numbers therefore needs the old script from git history.

## Notes

- Training data on disk is 7-channel; NDSI and NDWI bring it to the 9 the
  model takes. The loader checks this per image and names the offending file.
- Two tif vintages exist: the current set reads back as `(1, 128, 128, 7)`,
  an older one as `(128, 128, 8)`. The loader normalises both to channel-first.
- Reported test Dice is 0.92. Full ablation trail in `.agents/references/paper.md`.

## References

- `.agents/context/model.md` — architecture and the checkpoint contract
- `docs/MODEL_MANIFEST.md` — what each existing checkpoint is
