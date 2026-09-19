"""Read Landsat GeoTIFFs and DEMs off disk into numpy arrays.

Every raster comes back in HWC layout (height, width, channel) with negative
values and -inf clamped to zero, since those mark unmeasured pixels rather
than real reflectance.
"""

import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject


def reproject_raster(infp, outfp, dst_crs='EPSG:4326'):
    """Reproject a raster to `dst_crs`, writing to `outfp`.

    Landsat scenes arrive in UTM, which splits the globe into zones — so one
    glacier photographed on two dates can land in two different coordinate
    systems. Reprojecting to a common CRS is what lets a lat/long GLIMS
    polygon line up with the pixels.
    """
    with rasterio.open(infp) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.meta.copy()
        profile.update(crs=dst_crs, transform=transform, width=width,
                       height=height, nodata=0)

        with rasterio.open(outfp, 'w', **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0,
                )


def _clean(raster):
    """Clamp negative values to zero and move the band axis last.

    The comparison also catches -inf. NaN and +inf are deliberately left
    alone: they mean "no usable measurement", and callers decide how to fill
    them (final_areas.py substitutes the stack mean) rather than silently
    treating them as zero reflectance.
    """
    raster[raster < 0] = 0
    return np.rollaxis(raster, 0, 3)


def get_rasters(dir_path_rasters, file_name=None, filtered_file_path=None):
    """Read the GeoTIFFs in a directory into `{file_name: HWC array}`.

    Args:
        dir_path_rasters: directory to scan (one per glacier).
        file_name: read only this one file instead of the whole directory.
        filtered_file_path: CSV with a `file_name` column; only files listed
            there are read. This is how the metadata quality filter is applied
            before anything touches memory.
    """
    if file_name:
        names = [file_name]
    else:
        names = [f for f in next(os.walk(dir_path_rasters))[2] if f.endswith('.tif')]

    if filtered_file_path:
        allowed = set(pd.read_csv(filtered_file_path)["file_name"])
        names = [f for f in names if f in allowed]

    rasters = {}
    for name in names:
        with rasterio.open(os.path.join(dir_path_rasters, name)) as src:
            rasters[name] = _clean(src.read())
    return rasters


def get_dem(file_path_dem):
    """Read a DEM's first band as `{file_name: HWC array}`.

    Returned as a single-entry dict so it composes with `get_rasters` output.
    One DEM covers a glacier for every date, since terrain is static.
    """
    with rasterio.open(file_path_dem) as src:
        dem = _clean(src.read()[[0], :, :])
    return {os.path.basename(file_path_dem): dem}
