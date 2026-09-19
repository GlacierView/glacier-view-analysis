"""GlacierView — glacier surface-area estimation from Landsat imagery.

Public surface:

    from glacierview import config
    from glacierview.models import UNet, IN_CHANNELS, load_checkpoint
    from glacierview.preprocess import prepare_glacier_stack, add_spectral_indices

Command line: ``glacierview --help``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
