"""Inference: segment glaciers and derive surface areas."""

from glacierview.inference.predict import (
    areas_km2,
    build_inputs,
    predict_probabilities,
    segment_glacier,
)

__all__ = ["areas_km2", "build_inputs", "predict_probabilities", "segment_glacier"]
