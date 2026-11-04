"""Compare the Python port with reference results of the original MATLAB code.

    python make_data.py
    octave-cli --eval "run_reference('../../cidre/trunk/matlab', 'data/cells.mat', 'data/cells_ref.mat')"   # or MATLAB
    python compare.py cells

Reports (1) agreement of the pre-processed quantile stack, (2) agreement of
the final gain surface v between the two implementations, and (3) the error of
each implementation against the known ground-truth illumination.
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import cidre
from cidre._matlab import imresize
from cidre.model import load_stack
from cidre.preprocess import preprocess

data = Path(__file__).resolve().parent / "data"
logging.disable(logging.INFO)


def norm(v):
    return v / v.mean()


def compare(name: str) -> None:
    d = np.load(data / f"{name}.npz")
    ref = loadmat(data / f"{name}_ref.mat", squeeze_me=True)
    I, v_true, z_true = d["stack"], d["v"], d["z"]

    stack, max_i, _, _ = load_stack(I)
    pre = preprocess(stack, max_i)
    S_ref = np.moveaxis(ref["S_pre"], 2, 0)
    pre_diff = np.abs(pre.stack - S_ref).max() / np.ptp(S_ref)

    m = cidre.fit(I)

    vt = norm(imresize(v_true, m.v_small.shape, method="bilinear"))
    vp, vr = norm(m.v_small), norm(ref["v_small"])
    print(f"[{name}] {I.shape[0]} images {I.shape[1]}x{I.shape[2]}, working size {m.v_small.shape}")
    print(f"  pre-processed stack, max rel. difference : {pre_diff:.1e}")
    print(f"  v, Python vs MATLAB  mean |diff| {np.abs(vp - vr).mean():.4f}   max {np.abs(vp - vr).max():.4f}")
    print(f"  v error vs truth     MATLAB {np.abs(vr - vt).mean():.4f}   Python {np.abs(vp - vt).mean():.4f}")
    print(f"  z mean (truth {z_true.mean():.1f})   MATLAB {ref['z_small'].mean():.1f}   Python {m.z_small.mean():.1f}")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["iid", "cells", "few"]:
        compare(name)
