# Code rules

- The model is defined **once**, in `src/segmentation/inference/cnn.py`. Import `UNet` and
  `conv_block` from there; do not copy the classes into another file.
- Checkpoints are pickled modules, not state dicts. Renaming or reshaping a layer can stop an
  existing checkpoint loading. Old checkpoints reference `__main__.UNet` and `__main__.conv_block`,
  so a script that loads one must import both names into its own globals and run as `__main__`.
- Input channel count comes from `IN_CHANNELS` in `inference/cnn.py`. Do not hardcode it.
- Filenames are load-bearing: `{glims_id}_{date}_{L5|L7|L8}_C02_T1_L2_SR.tif`. Token 0 is the GLIMS
  ID, token 1 the date, token 2 the satellite used to pick the band map. Renaming a file silently
  changes which bands get read.
- `helpers/read.py` and `helpers/preprocess.py` are shared by both inference entry points. A change
  there affects them equally — that is intentional; don't fork them.
- Plots use the shared style `src/styles/ieee.mplstyle`.
- `uv run ruff check .` must pass. The rule set is pinned in `pyproject.toml`; notebooks carry
  documented per-file exemptions. `ruff format` is deliberately not enforced.
