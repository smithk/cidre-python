"""The CIDRE energy and its analytic gradient (port of ``cdr_objective``).

The variables are the gain surface ``v``, the (pivoted) offset surface ``b`` and
the zero-light point ``(zx, zy)``. The energy is

    E = E_fit + 10**lambda_v * E_vreg + 10**lambda_z * E_zero

* ``E_fit``  robust (Cauchy) fit of every pixel's quantile curve ``q`` to the
  global intensity distribution ``Q``: ``q ~ v * Q + b``.
* ``E_vreg`` multi-scale, scale-normalised Laplacian-of-Gaussian smoothness
  penalty on ``v`` (``symmetric`` boundary handling, as ``imfilter`` in MATLAB),
  ``sum_n ||h_n * v||^2 / K`` over ``K`` scales (see :func:`smoothness_operator`).
* ``E_zero`` asks all per-pixel lines to pass through one common zero-light
  point ``(zx, zy)``.

The MATLAB code kept ``zx`` and ``zy`` inside ``[zmin, zmax]`` with a steep
quadratic barrier; here the optimiser enforces those limits directly as bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ._matlab import fspecial_log

N_STANDARD = 200  # the number of quantiles the regularisation weights were tuned for


def log_sigmas(shape: tuple[int, int]) -> np.ndarray:
    """LoG scales used by the smoothness term: ``2 ** (-1 : max_exp)``."""
    max_exp = max(1.0, math.log2(math.floor(max(shape) / 50))) if max(shape) >= 50 else 1.0
    return 2.0 ** np.arange(-1, math.floor(max_exp) + 1)


def log_kernels(shape: tuple[int, int]) -> list[np.ndarray]:
    """Scale-normalised LoG kernels ``sigma**2 * fspecial('log', hsize, sigma)``."""
    kernels = []
    for s in log_sigmas(shape):
        hsize = 6 * math.ceil(s)
        if hsize % 2 == 0:
            hsize += 1
        kernels.append(s * s * fspecial_log(hsize, s))
    return kernels


def _mirror(idx: np.ndarray, n: int) -> np.ndarray:
    """Map possibly out-of-range indices with MATLAB's ``symmetric`` padding rule."""
    period = 2 * n
    m = np.mod(idx, period)
    return np.where(m < n, m, period - 1 - m)


def filter_operator(kernel: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    """Sparse matrix ``L`` with ``L @ v.ravel() == imfilter(v, kernel, 'symmetric').ravel()``."""
    R, C = shape
    kr, kc = kernel.shape
    pr, pc = kr // 2, kc // 2
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    rows, cols, vals = [], [], []
    out_index = (rr * C + cc).ravel()
    for i in range(kr):
        src_r = _mirror(rr + i - pr, R)
        for j in range(kc):
            w = kernel[i, j]
            if w == 0:
                continue
            src_c = _mirror(cc + j - pc, C)
            rows.append(out_index)
            cols.append((src_r * C + src_c).ravel())
            vals.append(np.full(R * C, w))
    L = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(R * C, R * C)
    )
    L.sum_duplicates()
    return L


def smoothness_operator(shape: tuple[int, int]) -> sp.csr_matrix:
    """``A`` such that ``E_vreg(v) = v @ A @ v = sum_n ||h_n * v||^2 / K``.

    A note on the ``1 / K`` normalisation over the ``K`` LoG scales: the MATLAB
    code evaluates the energy as ``sum_n ||(h_n * v) / K||^2`` (i.e. ``1 / K**2``)
    but its hand-written gradient is that of ``sum_n ||h_n * v||^2 / K``, ``K``
    times larger. The optimiser follows the gradient, so the regularisation
    strength CIDRE's default ``lambda_v`` values were tuned with is the ``1 / K``
    one. Using it here (with an exact gradient) reproduces the MATLAB models
    closely; see ``validation/README.md``.
    """
    kernels = log_kernels(shape)
    K = len(kernels)
    A = None
    for h in kernels:
        L = filter_operator(h, shape)
        term = (L.T @ L).tocsr()
        A = term if A is None else A + term
    return (A / K).tocsr()


@dataclass
class CidreProblem:
    """Everything the objective needs, in pivot space."""

    S: np.ndarray  # (Z, R, C) pivoted quantile stack
    Q: np.ndarray  # (Z,) pivoted global intensity distribution
    pivot_x: float
    pivot_y: np.ndarray  # (R, C)
    lambda_v: float  # log10 weight of the smoothness term
    lambda_z: float  # log10 weight of the zero-light term
    cauchy_w: float = 1.0
    A: sp.csr_matrix = field(default=None, repr=False)

    def __post_init__(self):
        if self.A is None:
            self.A = smoothness_operator(self.S.shape[1:])

    @property
    def shape(self) -> tuple[int, int]:
        return self.S.shape[1:]

    @property
    def npix(self) -> int:
        R, C = self.shape
        return R * C

    def pack(self, v: np.ndarray, b: np.ndarray, zx: float, zy: float) -> np.ndarray:
        return np.concatenate([np.ravel(v), np.ravel(b), [zx, zy]]).astype(float)

    def unpack(self, x: np.ndarray):
        n = self.npix
        return x[:n].reshape(self.shape), x[n : 2 * n].reshape(self.shape), float(x[-2]), float(x[-1])

    # ------------------------------------------------------------------
    def energy_terms(self, x: np.ndarray) -> dict[str, float]:
        """The individual (unweighted) energy terms, for diagnostics."""
        v, b, zx, zy = self.unpack(x)
        r = self.Q[:, None, None] * v + b - self.S
        w2 = self.cauchy_w**2
        fit = 0.5 * w2 * np.log1p(r * r / w2).sum() * (N_STANDARD / self.S.shape[0])
        vv = v.ravel()
        res0 = vv * (zx - self.pivot_x) + b.ravel() - (zy - self.pivot_y.ravel())
        return {"fit": float(fit), "vreg": float(vv @ (self.A @ vv)), "zero": float(res0 @ res0)}

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Energy and gradient of the robust, regularised problem (CIDRE's 2nd stage)."""
        n = self.npix
        v = x[:n]
        b = x[n : 2 * n]
        zx, zy = x[-2], x[-1]
        Z = self.S.shape[0]
        factor = N_STANDARD / Z
        lv, lz = 10.0**self.lambda_v, 10.0**self.lambda_z

        # robust fit term (Cauchy M-estimator)
        S = self.S.reshape(Z, n)
        w2 = self.cauchy_w**2
        r = self.Q[:, None] * v
        r += b
        r -= S  # residuals of q ~ v * Q + b
        d = r * r
        d *= 1.0 / w2
        d += 1.0
        E_fit = 0.5 * w2 * np.log(d).sum() * factor
        r /= d  # influence function psi(r) = r / (1 + r^2 / w^2)
        g_v = (self.Q @ r) * factor
        g_b = r.sum(axis=0) * factor

        # spatial smoothness of v (multi-scale LoG)
        Av = self.A @ v
        E_vreg = float(v @ Av)
        g_v += lv * 2.0 * Av

        # zero-light term: every line passes through (zx, zy)
        px = zx - self.pivot_x
        py = zy - self.pivot_y.ravel()
        res0 = v * px + b - py
        E_zero = float(res0 @ res0)
        g_v += lz * 2.0 * px * res0
        g_b += lz * 2.0 * res0
        g_zx = lz * 2.0 * float(v @ res0)
        g_zy = lz * -2.0 * float(res0.sum())

        E = E_fit + lv * E_vreg + lz * E_zero
        return E, np.concatenate([g_v, g_b, [g_zx, g_zy]])
