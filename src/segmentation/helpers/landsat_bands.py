"""Band name to array-index maps, one per Landsat satellite.

Landsat 5 and 7 share a layout. Landsat 8's sensor adds a coastal-aerosol band
at index 0, which shifts everything after it — so `swir` is index 4 on L5/L7
but index 5 on L8. `preprocess.get_common_bands` uses these maps to pull a
consistent set of channels whichever satellite a scene came from.

The `extra*` entries are padding, not real bands. `get_common_bands` builds its
selection mask with `np.isin(list(landsat_bands[sat]), common_bands)`, and that
mask is applied to the raster's band axis — so each map has to have exactly as
many entries as the GeoTIFF has bands (19). Do not trim them.
"""

_L5_L7_BANDS = {
    'blue': 0,
    'green': 1,
    'red': 2,
    'nir': 3,
    'swir': 4,
    'swir_2': 5,
    'atmos_opacity': 6,
    'cloud_qa': 7,
    'thermal': 8,
    **{f'extra{i}': 9 + i for i in range(10)},
}

_L8_BANDS = {
    'coastal_aerosol': 0,
    'blue': 1,
    'green': 2,
    'red': 3,
    'nir': 4,
    'swir': 5,
    'swir_2': 6,
    'qa_aerosol': 7,
    'thermal': 8,
    'atmospheric_transmittance': 9,
    'thermal_2': 10,
    'bitmap': 11,
    **{f'extra{i}': 11 + i for i in range(1, 8)},
}

landsat_bands = {
    'l5': _L5_L7_BANDS,
    'l7': _L5_L7_BANDS,
    'l8': _L8_BANDS,
}
