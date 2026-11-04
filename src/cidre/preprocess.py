"""Pre-processing of the image stack (port of ``cdr_loadImages`` / ``cdr_preprocessData``).

The stack is resized to a small working size, checked for sufficient intensity
information (entropy), optionally augmented with a scale-space resampling, sorted
per pixel and compressed to at most ``number_of_quantiles`` slices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from ._matlab import hist_counts, imresize, mround

log = logging.getLogger(__name__)

VALID_BIT_DEPTHS = (2**8, 2**12, 2**14, 2**16)


def working_size(image_size: tuple[int, int], target_num_pixels: int = 9400) -> tuple[int, int]:
    """Working size with roughly ``target_num_pixels`` pixels and the same aspect ratio."""
    r, c = image_size
    s = math.sqrt(target_num_pixels / (r * c))
    return int(mround(r * s)), int(mround(c * s))


def estimate_bit_depth(max_intensity: float) -> int:
    """Bit depth (as the max integer value, e.g. 4096) estimated from the brightest pixel."""
    if max_intensity > 2**14:
        return 2**16
    if max_intensity > 2**12:
        return 2**14
    if max_intensity > 2**8:
        return 2**12
    return 2**8


def stack_entropy(S: np.ndarray, bit_depth: int) -> float:
    """Shannon entropy (bits) of the intensity distribution of the whole stack."""
    if not np.all(np.isfinite(S)):
        raise ValueError("the input images contain non-finite values")
    p = hist_counts(S, bit_depth).astype(float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def scale_space_resampling(S: np.ndarray, entropy: float) -> np.ndarray:
    """Augment a stack with too little intensity information.

    When fewer images are supplied than empirically needed for the observed
    entropy, coarser octaves of the stack (resampled back to the working size)
    are appended so that each location borrows information from its
    neighbourhood. ``S`` has shape ``(N, R, C)``.
    """
    n, r1, c1 = S.shape
    a, b, c = 7.838e06, -1.948, 20.0  # fitted exponential, see the CIDRE paper
    n_required = a * math.exp(b * entropy) + c

    if n < n_required:
        log.warning(
            "fewer images than recommended (%d < %d) for the observed entropy %.2f; "
            "using scale-space resampling to compensate",
            n,
            mround(n_required),
            entropy,
        )
        alpha = 1.0 - n / n_required
    else:
        alpha = 0.0

    octaves = [S]
    r, c = r1, c1
    while r > 1 and c > 1:
        octaves.append(imresize(octaves[-1], scale=0.5))
        r, c = octaves[-1].shape[-2:]

    alpha = min(1.0, max(0.0, alpha))
    max_i = max(1, math.ceil(alpha * len(octaves)))
    if max_i == 1:
        return S

    log.info("applying scale-space resampling with %d octaves", max_i)
    return np.concatenate([imresize(octaves[i], (r1, c1)) for i in range(max_i)], axis=0)


def compress_stack(S: np.ndarray, number_of_quantiles: int = 200) -> np.ndarray:
    """Average runs of consecutive (sorted) slices down to ``number_of_quantiles`` slices."""
    z = S.shape[0]
    if z <= number_of_quantiles:
        return S
    zmin, zdiff = 1, z - 1
    lo = [1] + [mround(zmin + zdiff * (i - 1) / number_of_quantiles) + 1 for i in range(2, number_of_quantiles + 1)]
    hi = [mround(zmin + zdiff * i / number_of_quantiles) for i in range(1, number_of_quantiles + 1)]
    return np.stack([S[l - 1 : h].mean(axis=0) for l, h in zip(lo, hi)], axis=0)


@dataclass
class PreprocessResult:
    stack: np.ndarray  # (Z, R, C) sorted & compressed quantile stack
    bit_depth: int
    entropy: float
    num_images: int


def preprocess(
    S: np.ndarray,
    max_intensity: float,
    *,
    bit_depth: int | None = None,
    number_of_quantiles: int = 200,
) -> PreprocessResult:
    """Run the full pre-processing on a working-size stack ``S`` of shape ``(N, R, C)``."""
    if bit_depth is None:
        bit_depth = estimate_bit_depth(max_intensity)
        log.info("%d-bit images (estimated from max intensity %g)", int(math.log2(bit_depth)), max_intensity)
    elif bit_depth not in VALID_BIT_DEPTHS:
        raise ValueError("give bit_depth as the max integer value: 2**8, 2**12, 2**14 or 2**16")

    entropy = stack_entropy(S, bit_depth)
    log.info("entropy of the stack = %.2f", entropy)
    num_images = S.shape[0]
    S = scale_space_resampling(S, entropy)
    S = np.sort(S, axis=0)
    S = compress_stack(S, number_of_quantiles)
    return PreprocessResult(S, bit_depth, entropy, num_images)
