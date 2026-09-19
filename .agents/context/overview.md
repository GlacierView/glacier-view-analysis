# What this project is

A research pipeline that measures glacier surface-area change over time. It pulls Landsat 5/7/8 imagery
and NASADEM elevation from Google Earth Engine for glaciers identified by GLIMS ID, segments each image
with a ResNet-50-encoder U-Net (PyTorch), and converts the predicted masks into per-glacier area time
series (1984–present) plus GIFs.

Headline results and the paper's own numbers: `.agents/references/paper.md`.
Long-form narrative for a newcomer: `docs/ONBOARDING.md`.
