"""Building the CIDRE correction model and applying it (port of ``cdr_cidreModel`` / ``cdr_correct``)."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

import numpy as np
from scipy.optimize import minimize

from ._matlab import imresize, mround
from .io import PathLike, list_images, read_image, write_image
from .objective import CidreProblem
from .preprocess import preprocess, working_size

log = logging.getLogger(__name__)

CORRECTION_MODES = {
    "zero_light_preserved": 0,
    "dynamic_range_corrected": 1,
    "direct": 2,
}
_MODE_NAMES = {v: k for k, v in CORRECTION_MODES.items()}

ImageSource = Union[PathLike, Sequence[PathLike], np.ndarray, Sequence[np.ndarray]]


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------
@dataclass
class CidreModel:
    """An illumination model ``I_observed = v * I_true + z``.

    ``v`` is the multiplicative gain (vignetting) surface and ``z`` the additive
    zero-light (dark) surface, both at full image resolution. ``v_small`` and
    ``z_small`` are the surfaces at the working resolution the model was
    estimated at.
    """

    v: np.ndarray
    z: np.ndarray
    v_small: np.ndarray | None = None
    z_small: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(self.v.shape)

    # -- correction -------------------------------------------------------
    def correct(
        self,
        image: np.ndarray,
        mode: str | int = "zero_light_preserved",
        dtype: Any = "same",
    ) -> np.ndarray:
        """Correct one image (``(H, W)``) or a stack (``(N, H, W)``).

        Modes:

        * ``"zero_light_preserved"`` (0, default): ``(I - z) / v * mean(v) + mean(z)``;
          keeps the original intensity range *and* zero-light level.
        * ``"dynamic_range_corrected"`` (1): ``(I - z) / v * mean(v)``; keeps the
          intensity range but removes the zero-light offset.
        * ``"direct"`` (2): ``(I - z) / v``.

        By default the result has the input's dtype; integer outputs are rounded
        and saturated like MATLAB's ``cast``. Pass ``dtype=np.float32`` (or any
        dtype) to choose another output type, or ``None`` for float64.
        """
        img = np.asarray(image)
        if img.shape[-2:] != self.v.shape:
            raise ValueError(f"image size {img.shape[-2:]} does not match the model size {self.v.shape}")
        m = _mode_index(mode)
        out = (img.astype(float) - self.z) / self.v
        if m == 0:
            out = out * self.v.mean() + self.z.mean()
        elif m == 1:
            out = out * self.v.mean()
        target = img.dtype if (isinstance(dtype, str) and dtype == "same") else dtype
        return _cast(out, target)

    # -- persistence ------------------------------------------------------
    def save(self, path: PathLike) -> Path:
        """Save as ``.npz`` (NumPy) or ``.mat`` (readable by the MATLAB CIDRE tools)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".mat":
            from scipy.io import savemat

            model = {"method": "CIDRE", "v": self.v, "z": self.z}
            if self.v_small is not None:
                model["v_small"] = self.v_small
                model["z_small"] = self.z_small
            options = {k: (np.nan if v is None else v) for k, v in self.info.items() if _matlab_friendly(v)}
            savemat(path, {"model": model, "options": options}, do_compression=True)
        else:
            if path.suffix.lower() != ".npz":
                path = path.with_suffix(".npz")
            arrays = {"v": self.v, "z": self.z}
            if self.v_small is not None:
                arrays.update(v_small=self.v_small, z_small=self.z_small)
            np.savez_compressed(path, info=json.dumps(_jsonable(self.info)), **arrays)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "CidreModel":
        """Load a model saved by this package, by the MATLAB ``cidre`` (``cidre_model.mat``)
        or by the Fiji plugin (a folder holding ``cidre_model_v.csv`` etc.)."""
        path = Path(path)
        if path.is_dir():
            return cls._load_fiji(path)
        if path.suffix.lower() == ".mat":
            from scipy.io import loadmat

            d = loadmat(path, squeeze_me=True, struct_as_record=False)
            m = d["model"]
            get = lambda name: np.asarray(getattr(m, name), dtype=float) if hasattr(m, name) else None  # noqa: E731
            info = {}
            if "options" in d:
                opts = d["options"]
                for k in getattr(opts, "_fieldnames", []):
                    val = getattr(opts, k)
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    if isinstance(val, (int, float, str, list)):
                        info[k] = val
            return cls(get("v"), get("z"), get("v_small"), get("z_small"), info)
        with np.load(path, allow_pickle=False) as d:
            info = json.loads(str(d["info"])) if "info" in d else {}
            small = ("v_small" in d) and ("z_small" in d)
            return cls(
                d["v"].astype(float),
                d["z"].astype(float),
                d["v_small"].astype(float) if small else None,
                d["z_small"].astype(float) if small else None,
                info,
            )

    @classmethod
    def _load_fiji(cls, folder: Path) -> "CidreModel":
        def csv(name):
            f = folder / name
            return np.loadtxt(f, delimiter=",", ndmin=2) if f.exists() else None

        v, z = csv("cidre_model_v.csv"), csv("cidre_model_z.csv")
        if v is None or z is None:
            raise FileNotFoundError(f"{folder} does not contain cidre_model_v.csv / cidre_model_z.csv")
        return cls(v, z, csv("cidre_model_v_small.csv"), csv("cidre_model_z_small.csv"), {"source": "fiji"})


# ----------------------------------------------------------------------------
# estimation
# ----------------------------------------------------------------------------
def default_lambda_v(num_images: int) -> float:
    """More smoothing for few images: 9.5 at N = 0, linearly down to 6 at N >= 200."""
    nmax, l0, l1 = 200, 9.5, 6.0
    return l0 + (l1 - l0) / nmax * num_images if num_images < nmax else l1


def estimate_Q(S: np.ndarray, q_percent: float = 0.25) -> np.ndarray:
    """Robust mean of the per-pixel quantile curves, taken over the pixels whose
    mean intensity lies in the central ``q_percent`` of all pixels."""
    Z, R, C = S.shape
    n = R * C
    k = mround(q_percent * n)
    mean_surf = S.mean(axis=0).ravel(order="F")  # MATLAB column-major order
    inds = np.argsort(mean_surf, kind="stable")
    lo = max(mround(n / 2) - mround(k / 2) - 1, 0)
    hi = min(mround(n / 2) + mround(k / 2), n)
    r, c = np.unravel_index(inds[lo:hi], (R, C), order="F")
    return S[:, r, c].mean(axis=1)


def _least_squares_lines(S: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel least-squares fit of ``q = v * Q + b`` (CIDRE's first stage, solved exactly)."""
    Qm = Q.mean()
    dQ = Q - Qm
    var = float(dQ @ dQ)
    if var <= 0:
        raise ValueError("the images carry no intensity variation; cannot estimate a correction")
    qm = S.mean(axis=0)
    v = np.tensordot(dQ, S - qm, axes=(0, 0)) / var
    b = qm - v * Qm
    return v, b


def _standard_error(v, b, S, Q) -> float:
    Z = S.shape[0]
    if Z <= 2:
        raise ValueError("at least 3 distinct quantiles are needed; provide more images")
    res = S - (b + Q[:, None, None] * v)
    return float(np.sqrt((res * res).sum(axis=0) / (Z - 2)).mean())


def estimate_model(
    S: np.ndarray,
    *,
    image_size: tuple[int, int],
    num_images: int,
    lambda_v: float | None = None,
    lambda_z: float = 0.5,
    q_percent: float = 0.25,
    z_limits: tuple[float, float] | None = None,
    max_iterations: int = 500,
    max_evaluations: int = 1000,
) -> CidreModel:
    """Estimate ``v`` and ``z`` from a pre-processed quantile stack ``S`` ``(Z, R, C)``."""
    t0 = time.perf_counter()
    S = np.array(S, dtype=float)
    Z, R, C = S.shape
    if lambda_v is None:
        lambda_v = default_lambda_v(num_images)

    stack_min = float(S.min())
    if z_limits is not None:
        zmin, zmax = float(z_limits[0]), float(z_limits[1])
        if zmax < zmin:
            raise ValueError("z_limits must be given as (min, max)")
        z0 = 0.5 * (zmin + zmax)
    else:
        zmin, zmax = 0.0, max(stack_min, 0.0)
        z0 = 0.85 * zmax

    Q = estimate_Q(S, q_percent)

    # move to the pivot space: shift the origin to the median datum
    mid = mround(Z / 2) - 1
    pivot_x = float(Q[mid])
    Q = Q - pivot_x
    pivot_y = S[mid].copy()
    S -= pivot_y

    log.info(
        "optimising with lambda_v=%.2f lambda_z=%.2f q_percent=%.2f z_limits=[%g, %g]",
        lambda_v, lambda_z, q_percent, zmin, zmax,
    )

    # stage 1: ordinary least squares, independently for every pixel
    v1, b1 = _least_squares_lines(S, Q)
    cauchy_w = _standard_error(v1, b1, S, Q)

    # stage 2: robust regression with smoothness and zero-light terms
    problem = CidreProblem(S, Q, pivot_x, pivot_y, lambda_v, lambda_z, cauchy_w)
    x1 = problem.pack(v1, b1, z0, z0)
    n = problem.npix
    bounds = [(None, None)] * (2 * n) + [(zmin, zmax), (zmin, zmax)]
    res = minimize(
        problem,
        x1,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iterations, "maxfun": max_evaluations, "maxcor": 100, "ftol": 0.0, "gtol": 0.0},
    )
    v, b_pivoted, zx, zy = problem.unpack(res.x)

    # back to the original space; z is the line value at the zero-light intensity zx
    b = pivot_y + b_pivoted - pivot_x * v
    z = b + zx * v

    elapsed = time.perf_counter() - t0
    log.info("model estimated in %.2fs (%d iterations, %d evaluations)", elapsed, res.nit, res.nfev)

    info = {
        "method": "CIDRE",
        "lambda_v": float(lambda_v),
        "lambda_z": float(lambda_z),
        "q_percent": float(q_percent),
        "z_limits": [zmin, zmax],
        "zero_light_point": [zx, zy],
        "num_images": int(num_images),
        "num_quantiles": int(Z),
        "working_size": [int(R), int(C)],
        "image_size": [int(image_size[0]), int(image_size[1])],
        "cauchy_width": cauchy_w,
        "iterations": int(res.nit),
        "evaluations": int(res.nfev),
        "energy": float(res.fun),
    }
    return CidreModel(
        v=imresize(v, image_size, method="bilinear"),
        z=imresize(z, image_size, method="bilinear"),
        v_small=v,
        z_small=z,
        info=info,
    )


def load_stack(
    source: ImageSource,
    *,
    target_num_pixels: int = 9400,
    image_size: tuple[int, int] | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, float, tuple[int, int], list[Path] | None]:
    """Read ``source`` and resize every image to the working size.

    Returns the working-size stack ``(N, R, C)``, the maximum raw intensity,
    the original image size and (for file sources) the list of files.
    """
    if isinstance(source, np.ndarray) or (
        isinstance(source, Sequence) and len(source) > 0 and isinstance(source[0], np.ndarray)
    ):
        arr = np.asarray(source)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError("pass images as an (N, H, W) array or a sequence of 2-D arrays")
        size = tuple(image_size) if image_size is not None else arr.shape[1:]
        wsize = working_size(size, target_num_pixels)
        stack = np.stack([imresize(a, wsize) for a in arr])
        return stack, float(arr.max()), size, None

    files = list_images(source)
    first = read_image(files[0])
    size = first.shape
    wsize = working_size(size, target_num_pixels)
    log.info("reading %d images", len(files))

    def work(f):
        img = first if f == files[0] else read_image(f)
        if img.shape != size:
            raise ValueError(f"{f} has size {img.shape}, expected {size}: all images must share one size")
        return imresize(img, wsize), float(img.max())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, files))
    stack = np.stack([r[0] for r in results])
    return stack, max(r[1] for r in results), size, files


def fit(
    images: ImageSource,
    *,
    lambda_v: float | None = None,
    lambda_z: float = 0.5,
    q_percent: float = 0.25,
    z_limits: tuple[float, float] | None = None,
    bit_depth: int | None = None,
    max_iterations: int = 500,
    image_size: tuple[int, int] | None = None,
    target_num_pixels: int = 9400,
    number_of_quantiles: int = 200,
    workers: int | None = None,
) -> CidreModel:
    """Learn a CIDRE illumination model from a collection of images.

    Parameters
    ----------
    images
        A folder, glob pattern (``"plate/*_w2.tif"``), list of files, an
        ``(N, H, W)`` array or a sequence of 2-D arrays. Use images of a single
        channel; more images give a more robust model.
    lambda_v
        Spatial regularisation (log10). Larger values (e.g. 9.5) give a smoother
        gain surface ``v``. Default: 6 for 200+ images, more for fewer images.
    lambda_z
        Zero-light regularisation (log10). Larger values (e.g. 3) enforce a more
        uniform ``z`` surface.
    q_percent
        Fraction of pixels used for the robust estimate of the underlying
        intensity distribution ``Q``.
    z_limits
        Known ``(min, max)`` limits of the zero-light level, e.g. from dark frames.
    bit_depth
        Max integer value of the images (``2**8``, ``2**12``, ``2**14`` or
        ``2**16``); estimated from the data when omitted.
    max_iterations
        Maximum L-BFGS iterations.
    image_size
        Size of the images the model is produced for; only needed when the
        passed array has already been downsampled.
    """
    stack, max_i, size, _ = load_stack(images, target_num_pixels=target_num_pixels, image_size=image_size, workers=workers)
    pre = preprocess(stack, max_i, bit_depth=bit_depth, number_of_quantiles=number_of_quantiles)
    model = estimate_model(
        pre.stack,
        image_size=size,
        num_images=pre.num_images,
        lambda_v=lambda_v,
        lambda_z=lambda_z,
        q_percent=q_percent,
        z_limits=z_limits,
        max_iterations=max_iterations,
    )
    model.info.update(bit_depth=int(pre.bit_depth), entropy=pre.entropy)
    return model


def correct_files(
    model: CidreModel,
    files: Iterable[PathLike],
    destination: PathLike,
    mode: str | int = "zero_light_preserved",
    workers: int | None = None,
) -> list[Path]:
    """Correct image files and write them, under their original names, to ``destination``."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    def work(f):
        f = Path(f)
        out = destination / f.name
        if out.resolve() == f.resolve():
            raise ValueError(f"refusing to overwrite the source image {f}")
        write_image(out, model.correct(read_image(f), mode))
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, list(files)))


def run(
    source: PathLike | Sequence[PathLike],
    destination: PathLike | None = None,
    *,
    correction_mode: str | int = "zero_light_preserved",
    workers: int | None = None,
    **fit_kwargs,
) -> CidreModel:
    """The classic CIDRE workflow: learn a model from ``source`` and, if a
    ``destination`` is given, save ``cidre_model.mat`` there along with the
    corrected images (same file names)."""
    files = list_images(source)
    model = fit(files, workers=workers, **fit_kwargs)
    if destination is not None:
        destination = Path(destination)
        model.info["correction_mode"] = _MODE_NAMES[_mode_index(correction_mode)]
        model.save(destination / "cidre_model.mat")
        correct_files(model, files, destination, correction_mode, workers)
        log.info("wrote %d corrected images to %s", len(files), destination)
    return model


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _mode_index(mode: str | int) -> int:
    if isinstance(mode, (int, np.integer)):
        if int(mode) in _MODE_NAMES:
            return int(mode)
    elif isinstance(mode, str):
        key = mode.lower().replace("-", "_").replace(" ", "_")
        aliases = {"zero_light": 0, "zero_light_preserving": 0, "dynamic_range": 1, "intensity_range_preserving": 1}
        if key in CORRECTION_MODES:
            return CORRECTION_MODES[key]
        if key in aliases:
            return aliases[key]
    raise ValueError(f"unknown correction mode {mode!r}; use one of {list(CORRECTION_MODES)} or 0/1/2")


def _cast(x: np.ndarray, dtype) -> np.ndarray:
    if dtype is None:
        return x
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        x = np.nan_to_num(x, nan=0.0)
        x = np.sign(x) * np.floor(np.abs(x) + 0.5)  # MATLAB rounding
        return np.clip(x, info.min, info.max).astype(dtype)
    if dtype == np.bool_:
        return x > 0.5
    return x.astype(dtype)


def _jsonable(d: dict) -> dict:
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [conv(i) for i in v]
        return v

    return {k: conv(v) for k, v in d.items()}


def _matlab_friendly(v) -> bool:
    return v is None or isinstance(v, (int, float, str, np.number)) or (
        isinstance(v, (list, tuple)) and all(isinstance(i, (int, float)) for i in v)
    )
