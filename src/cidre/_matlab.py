"""Small, faithful re-implementations of the MATLAB built-ins CIDRE relies on.

The original CIDRE code leans on MATLAB-specific behaviour (``round`` rounding
halves away from zero, ``imresize`` antialiasing, ``fspecial('log')``,
``hist`` bin edges). Reproducing these exactly keeps the Python port's
pre-processing numerically aligned with the MATLAB implementation, which in
turn keeps the empirically tuned defaults (``lambda_v``, ``lambda_z``,
``q_percent`` ...) meaningful.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

__all__ = ["mround", "imresize", "imresize_matrix", "fspecial_log", "hist_counts"]


def mround(x):
    """MATLAB ``round``: halves are rounded away from zero (NumPy rounds to even)."""
    x = np.asarray(x, dtype=float)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return int(out) if out.ndim == 0 else out


# --------------------------------------------------------------------------
# imresize
# --------------------------------------------------------------------------
def _cubic(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    return (1.5 * ax3 - 2.5 * ax2 + 1.0) * (ax <= 1) + (
        -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0
    ) * ((ax > 1) & (ax <= 2))


def _triangle(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) * ((x >= -1) & (x < 0)) + (1.0 - x) * ((x >= 0) & (x <= 1))


_KERNELS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float]] = {
    "bicubic": (_cubic, 4.0),
    "bilinear": (_triangle, 2.0),
}

Boundary = Literal["symmetric", "replicate"]


def imresize_matrix(
    in_length: int,
    out_length: int,
    scale: float | None = None,
    method: str = "bicubic",
    antialiasing: bool = True,
    boundary: Boundary = "symmetric",
) -> np.ndarray:
    """Dense ``(out_length, in_length)`` matrix applying MATLAB's 1-D resize.

    ``scale`` defaults to ``out_length / in_length`` (MATLAB's behaviour when an
    output size is requested). ``boundary`` selects how taps falling outside the
    signal are handled: ``"symmetric"`` mirrors them (current MATLAB releases),
    ``"replicate"`` clamps them to the edge (older releases, and the CIDRE
    ImageJ plugin).
    """
    kernel, width = _KERNELS[method]
    if scale is None:
        scale = out_length / in_length
    if scale < 1 and antialiasing:
        h = lambda t: scale * kernel(scale * t)  # noqa: E731
        width = width / scale
    else:
        h = kernel

    x = np.arange(1, out_length + 1, dtype=float)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - width / 2.0)
    taps = int(np.ceil(width)) + 2
    idx = left[:, None] + np.arange(taps)[None, :]  # 1-based, may be out of range
    w = h(u[:, None] - idx)
    w = w / w.sum(axis=1, keepdims=True)

    if boundary == "symmetric":
        aux = np.concatenate([np.arange(1, in_length + 1), np.arange(in_length, 0, -1)])
        idx = aux[np.mod(idx - 1, aux.size).astype(int)]
    elif boundary == "replicate":
        idx = np.clip(idx, 1, in_length)
    else:  # pragma: no cover - guarded by the type
        raise ValueError(f"unknown boundary mode {boundary!r}")

    M = np.zeros((out_length, in_length))
    rows = np.repeat(np.arange(out_length), taps)
    np.add.at(M, (rows, idx.astype(int).ravel() - 1), w.ravel())
    return M


def imresize(
    img: np.ndarray,
    output_size: tuple[int, int] | None = None,
    *,
    scale: float | None = None,
    method: str = "bicubic",
    antialiasing: bool = True,
    boundary: Boundary = "symmetric",
) -> np.ndarray:
    """MATLAB-compatible ``imresize`` acting on the last two axes of ``img``.

    Give either ``output_size=(rows, cols)`` or a scalar ``scale`` (in which
    case the output size is ``ceil(scale * size)`` as in MATLAB). Leading axes
    (e.g. a stack of images ``(N, H, W)``) are resized independently.
    """
    img = np.asarray(img, dtype=float)
    in_r, in_c = img.shape[-2:]
    if (output_size is None) == (scale is None):
        raise ValueError("give exactly one of output_size or scale")
    if scale is not None:
        out_r, out_c = int(np.ceil(scale * in_r)), int(np.ceil(scale * in_c))
        sr = sc = float(scale)
    else:
        out_r, out_c = int(output_size[0]), int(output_size[1])
        sr, sc = out_r / in_r, out_c / in_c

    Mr = imresize_matrix(in_r, out_r, sr, method, antialiasing, boundary)
    Mc = imresize_matrix(in_c, out_c, sc, method, antialiasing, boundary)
    # (out_r, in_r) @ (..., in_r, in_c) @ (in_c, out_c)
    return np.matmul(np.matmul(Mr, img), Mc.T)


# --------------------------------------------------------------------------
# fspecial('log')
# --------------------------------------------------------------------------
def fspecial_log(size: int, sigma: float) -> np.ndarray:
    """MATLAB ``fspecial('log', size, sigma)`` (Laplacian of Gaussian, zero sum)."""
    half = (size - 1) / 2.0
    ax = np.arange(-half, half + 1)
    x, y = np.meshgrid(ax, ax)
    std2 = sigma * sigma
    r2 = x * x + y * y
    h = np.exp(-r2 / (2.0 * std2))
    h[h < np.finfo(float).eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h = h / s
    h1 = h * (r2 - 2.0 * std2) / (std2 * std2)
    return h1 - h1.sum() / h1.size


# --------------------------------------------------------------------------
# hist(y, 1:nbins)
# --------------------------------------------------------------------------
def hist_counts(values: np.ndarray, nbins: int) -> np.ndarray:
    """Counts of MATLAB ``hist(values, 1:nbins)``.

    Bin centres are ``1..nbins``; edges sit halfway between centres, the outer
    bins are open-ended and bins are right-closed, ``(a, b]``, as in MATLAB.
    """
    v = np.asarray(values, dtype=float).ravel()
    idx = np.clip(np.ceil(v - 0.5) - 1, 0, nbins - 1).astype(np.int64)
    return np.bincount(idx, minlength=nbins)
