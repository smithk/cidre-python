"""Generate the synthetic validation datasets (``data/<name>.npz`` for Python,
``data/<name>.mat`` with an R x C x N stack for MATLAB/Octave).

* ``iid``   200 images whose per-pixel intensity distributions are identical up
            to the illumination model (CIDRE's assumption holds exactly);
            tilted zero-light surface.
* ``cells`` 120 images of sparse "cells" on a fluorescent background; tilted
            zero-light surface.
* ``few``   25 small images, low intensity information: exercises the
            scale-space resampling path.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synthetic import make_collection, true_surfaces  # noqa: E402

out = Path(__file__).resolve().parent / "data"
out.mkdir(exist_ok=True)


def save(name, stack, v, z):
    np.savez(out / f"{name}.npz", stack=stack, v=v, z=z)
    savemat(out / f"{name}.mat", {"stack": np.transpose(stack, (1, 2, 0)).astype(float)})


rng = np.random.default_rng(0)
shape = (120, 160)
v, z = true_surfaces(shape, 100, 30)
L = rng.gamma(2.0, 400.0, (200,) + shape)
L = np.stack([gaussian_filter(l, 2) for l in L])
L = L / L.mean() * 800
I = np.clip(np.round(v * L + z + 3 * rng.standard_normal(L.shape)), 0, 65535).astype(np.uint16)
save("iid", I, v, z)

imgs, _, v2, z2 = make_collection(n=120, z_tilt=30)
save("cells", imgs, v2, z2)

imgs3, _, v3, z3 = make_collection(n=25, shape=(100, 130), seed=3, z_tilt=0)
save("few", imgs3, v3, z3)
print(f"wrote datasets to {out}")
