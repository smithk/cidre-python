"""CIDRE: retrospective illumination correction for optical microscopy.

A Python implementation of

    K. Smith, Y. Li, F. Piccinini et al. "CIDRE: an illumination-correction
    method for optical microscopy". Nature Methods 12, 404-406 (2015).
    doi:10.1038/nmeth.3323

Typical use::

    import cidre

    model = cidre.fit("plate1/*_w1.tif")  # learn v (gain) and z (zero light)
    corrected = model.correct(image)      # correct any image of that channel
    model.save("w1_model.npz")

or, for the classic folder-to-folder workflow::

    cidre.run("plate1/*_w1.tif", "plate1_corrected/")
"""

from .model import (
    CORRECTION_MODES,
    CidreModel,
    correct_files,
    default_lambda_v,
    estimate_model,
    fit,
    load_stack,
    run,
)
from .preprocess import preprocess, working_size

__version__ = "1.0.0"

__all__ = [
    "CORRECTION_MODES",
    "CidreModel",
    "correct_files",
    "default_lambda_v",
    "estimate_model",
    "fit",
    "load_stack",
    "preprocess",
    "run",
    "working_size",
    "__version__",
]
