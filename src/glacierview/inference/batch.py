"""Segment every glacier in the landing zone and write area time series.

Aggregate CSVs are rewritten after each glacier, so a run that dies partway
still leaves usable output. Failures are logged with a traceback and counted
rather than aborting the run — with hundreds of glaciers, one bad scene
should not cost the whole batch. The tally at the end is the thing to read:
previously this loop could fail on every glacier and still look successful.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glacierview import config
from glacierview.inference import render
from glacierview.inference.predict import AREA_VARIANTS, segment_glacier
from glacierview.log import get_logger

logger = get_logger(__name__)

#: Monthly index the annual means are merged onto.
DATE_INDEX = pd.date_range(start="1979-01-01", end="2024-01-01", freq="MS")

CSV_NAMES = {
    "raw": "areas_no_threshold.csv",
    "thresholded": "areas_05_thresh.csv",
    "binary": "areas_binary_05.csv",
}


def to_annual_means(areas: list[float], dates, glims_id: str) -> pd.DataFrame:
    """Collapse a per-date series to one value per year, stamped mid-year.

    The mid-year offset makes each annual value plot at the centre of the
    period it summarises rather than at January.
    """
    series = pd.DataFrame({glims_id: areas}, index=pd.DatetimeIndex(dates))
    annual = series.groupby(pd.PeriodIndex(series.index, freq="Y"))[glims_id].mean()
    annual.index = annual.index.to_timestamp() + pd.DateOffset(months=6)
    return annual.to_frame().rename_axis("Dates").reset_index()


def run(
    model,
    device,
    out_dir: Path,
    data_label: str | None = None,
    glims_ids: list[str] | None = None,
    write_gifs: bool = True,
) -> dict:
    """Process the landing zone. Returns a summary dict."""
    landsat = config.landsat_dir(data_label)
    if not landsat.is_dir():
        raise FileNotFoundError(f"no landing zone at {landsat}")

    if glims_ids is None:
        glims_ids = sorted(
            d.name for d in landsat.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
    if not glims_ids:
        raise FileNotFoundError(f"{landsat} contains no glacier directories")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_glacier_dir = out_dir / "glacier_areas"
    per_glacier_dir.mkdir(exist_ok=True)

    aggregates = {
        name: pd.DataFrame({"Dates": DATE_INDEX}) for name in AREA_VARIANTS
    }
    failures: list[str] = []

    for i, glims_id in enumerate(glims_ids, start=1):
        try:
            result = segment_glacier(
                model, glims_id, device, data_label, variants=tuple(AREA_VARIANTS)
            )
        except Exception:
            failures.append(glims_id)
            logger.exception("[%d/%d] failed: %s", i, len(glims_ids), glims_id)
            continue

        for name, areas in result["areas"].items():
            frame = to_annual_means(areas, result["dates"], glims_id)
            aggregates[name] = aggregates[name].merge(frame, how="left", on="Dates")
            aggregates[name].to_csv(out_dir / CSV_NAMES[name], index=False)

        pd.DataFrame(
            {glims_id: result["areas"]["raw"]},
            index=pd.DatetimeIndex(result["dates"]),
        ).rename_axis("Dates").to_csv(per_glacier_dir / f"{glims_id}_areas.csv")

        if write_gifs:
            render.write_gif(
                gif_path=out_dir / "gifs" / f"{glims_id}_final.gif",
                scratch_dir=out_dir / "tmp" / glims_id,
                images=result["smoothed"],
                probabilities=result["probabilities"],
                file_names=result["file_names"],
                stride=10,
                overlay=True,
            )
        logger.info("[%d/%d] %s", i, len(glims_ids), glims_id)

    succeeded = len(glims_ids) - len(failures)
    logger.info("%d/%d glaciers succeeded", succeeded, len(glims_ids))
    if failures:
        logger.warning("%d failed: %s", len(failures), ", ".join(failures))
    return {"total": len(glims_ids), "succeeded": succeeded, "failures": failures}
