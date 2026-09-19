## The paper

The work is written up as an IEEE journal submission: **"A Large Scale Analysis of Mountain Glacier
Shrinkage Using Convolutional Neural Networks"** — Matthew Waismann (Amazon), Somansh Budhwar, and
Armin Schwartzman (Halıcıoğlu Data Science Institute, UC San Diego; `armins@ucsd.edu`). The LaTeX
source is not in this repo. Headline claims:

- Trained on **7,174** glacier contours + images, one time point each; **Dice 0.92** on the test set.
- Inference over **505,835 images for 1,083 glaciers**, 1984–2023.
- Overall glacier surface-area decline of **0.2% ± 0.01% per year**, negative and significant in all
  five regions.

Per-region relative area change per year (log-linear regression):

| Region | Δ/yr | Std err |
|---|---|---|
| Europe | −0.00256 | 0.00019 |
| South America | −0.00256 | 0.00029 |
| Caucasus | −0.00167 | 0.00025 |
| Asia | −0.00163 | 0.00015 |
| North America | −0.00155 | 0.00017 |
| **Total** | **−0.00199** | **0.00010** |

Cumulative 1985–2022 declines from the bootstrap analysis: Europe 16.2%, Caucasus 9.1%, Asia 7.1%,
South America 2.6%, North America 2.3%.

The GLIMS funnel, with the paper's numbers: **328,115** glaciers in the snapshot → **170,363** after
"outline drawn after 2005" → **122,212** after excluding Arctic/Antarctic regions → **18,093** after
the 1–50 km² area filter. That last number matches the repo exactly.

The paper is a work in progress and carries `\editcom{}` notes; two of its own numbers are stale
(the conclusion says "a dataset of 10,443 glaciers", which is really the 10,447-row
`training_data_set.csv`, and its body text says 10 input channels where its channel figure says 9 —
the figure is right, see "Model contract").

`docs/low_level_design.md` is the design doc — read it before changing anything in the data extraction,
metadata, or pre-inference filtering steps, since it records *why* the filter thresholds are what they
are and what the known tech debt is. Its concrete names (Glue tables, model band order) have drifted
from the code; where they disagree, the "Data" and "Model contract" sections below were checked against
the live account and the source.
