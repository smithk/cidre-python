"""Finding, reading and writing microscopy images."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

PathLike = Union[str, os.PathLike]

IMAGE_EXTENSIONS = (".bmp", ".gif", ".jpg", ".jpeg", ".tif", ".tiff", ".png")
_TIFF = (".tif", ".tiff")


def list_images(source: PathLike | Sequence[PathLike]) -> list[Path]:
    """Resolve ``source`` to a sorted list of image files.

    ``source`` may be a folder (every supported image in it is used), a glob
    pattern such as ``"plate1/*_w1.tif"``, a single file, or a list of files.
    """
    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        if p.is_dir():
            files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        elif any(ch in str(source) for ch in "*?["):
            files = [Path(f) for f in glob.glob(str(source)) if Path(f).is_file()]
        elif p.is_file():
            files = [p]
        else:
            raise FileNotFoundError(f"no such file, folder or pattern: {source}")
    else:
        files = [Path(f) for f in source]
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"no images found in {source}")
    return files


def read_image(path: PathLike) -> np.ndarray:
    """Read a single monochrome image, keeping its native dtype."""
    path = Path(path)
    if path.suffix.lower() in _TIFF:
        import tifffile

        img = tifffile.imread(path)
    else:
        import imageio.v3 as iio

        img = iio.imread(path)
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(
            f"{path} is not a single-channel 2-D image (shape {img.shape}). CIDRE corrects "
            "monochrome images: split multi-channel images and correct each channel separately."
        )
    return img


def write_image(path: PathLike, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in _TIFF:
        import tifffile

        tifffile.imwrite(path, img)
    else:
        import imageio.v3 as iio

        iio.imwrite(path, img)


def iter_images(files: Iterable[PathLike]) -> Iterable[np.ndarray]:
    for f in files:
        yield read_image(f)
