"""Synthetic microscopy-like image collections with a known illumination model."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def true_surfaces(shape=(180, 240), z_level=100.0, z_tilt=0.0):
    """A smooth, off-centre vignetting gain ``v`` and a zero-light surface ``z``."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = 0.45 * H, 0.55 * W
    r2 = ((yy - cy) / H) ** 2 + ((xx - cx) / W) ** 2
    v = 1.15 - 1.6 * r2  # ~1.15 at the centre, ~0.55 in the far corners
    z = z_level + z_tilt * (xx / W - 0.5)
    return v, z


def make_collection(n=120, shape=(180, 240), seed=0, z_level=100.0, z_tilt=0.0, noise=4.0, dtype=np.uint16):
    """``n`` images of a fluorescent background with randomly placed cells,
    distorted as ``I = v * L + z`` plus sensor noise."""
    rng = np.random.default_rng(seed)
    H, W = shape
    v, z = true_surfaces(shape, z_level, z_tilt)
    images = np.empty((n, H, W), dtype=dtype)
    truth = np.empty((n, H, W))
    for k in range(n):
        L = 250.0 + 30.0 * rng.standard_normal((H, W))
        L = gaussian_filter(L, 1.0)
        cells = np.zeros((H, W))
        m = rng.integers(15, 40)
        ys, xs = rng.integers(0, H, m), rng.integers(0, W, m)
        cells[ys, xs] = rng.uniform(800, 3000, m)
        cells = gaussian_filter(cells, 4.0) * 2 * np.pi * 16
        L = L + cells
        truth[k] = L
        I = v * L + z + noise * rng.standard_normal((H, W))
        info = np.iinfo(dtype)
        images[k] = np.clip(np.round(I), info.min, info.max).astype(dtype)
    return images, truth, v, z
