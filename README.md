# CIDRE for Python

CIDRE is a retrospective illumination correction method for optical microscopy. It learns the
illumination distortion of a microscope directly from a collection of images, as a gain
(vignetting) surface **v** and a zero-light (dark offset) surface **z**,

    I_observed(x) = v(x) * I_true(x) + z(x)

and then corrects every image of that collection. Larger collections give more robust models.

This package is a Python implementation of the method described in

> K. Smith, Y. Li, F. Piccinini et al.
> [CIDRE: an illumination-correction method for optical microscopy](https://doi.org/10.1038/nmeth.3323).
> *Nature Methods* 12, 404-406 (2015).

It succeeds the MATLAB implementation and ImageJ plugin in
[smithk/cidre](https://github.com/smithk/cidre), which remain available. It needs only NumPy, SciPy,
tifffile and imageio.

## Install

```bash
pip install "git+https://github.com/smithk/cidre-python"
```

or, from a clone, `pip install .`. This installs the `cidre` Python package and command. Install from
GitHub as shown: `pip install cidre` from PyPI gets an unrelated project of the same name.

## Use

Correct a folder (or glob) of images in one step. Use images of **one channel** per run; correct
each channel separately.

```bash
cidre run "plate1/*_w1.tif" -o plate1_corrected/
```

This writes the corrected images (same names, same bit depth) and `cidre_model.mat` to the output
folder. To learn a model once and apply it to other images:

```bash
cidre fit "plate1/*_w1.tif" -o w1_model.npz
cidre correct "plate2/*_w1.tif" --model w1_model.npz -o plate2_corrected/
```

From Python:

```python
import cidre

model = cidre.fit("plate1/*_w1.tif")          # a folder, glob, list of files or an (N, H, W) array
corrected = model.correct(image)               # one (H, W) image or an (N, H, W) stack
corrected = model.correct(image, mode="direct", dtype="float32")
model.save("w1_model.npz")                     # or .mat
model = cidre.CidreModel.load("w1_model.npz")  # also loads MATLAB cidre_model.mat and Fiji model folders

model.v, model.z                               # the illumination surfaces at full resolution
```

### Correction modes

| mode | formula | keeps |
|---|---|---|
| `zero_light_preserved` (default) | `(I - z) / v * mean(v) + mean(z)` | intensity range and zero-light level |
| `dynamic_range_corrected` | `(I - z) / v * mean(v)` | intensity range, zero light removed |
| `direct` | `(I - z) / v` | |

Integer images are rounded and saturated back to their own type; pass `dtype` to `correct` for
floating-point output.

### Parameters

| Python / CLI | default | effect |
|---|---|---|
| `lambda_v` / `--lambda-v` | 6 for 200+ images, up to 9.5 for very few | log10 weight of the smoothness of v; higher is smoother |
| `lambda_z` / `--lambda-z` | 0.5 | log10 weight of the zero-light term; higher gives a more uniform z |
| `z_limits` / `--z-limits MIN MAX` | 0 to the stack minimum | known limits of the zero-light level, e.g. from dark frames |
| `q_percent` / `--q-percent` | 0.25 | fraction of pixels used for the robust estimate of the intensity distribution |
| `bit_depth` / `--bit-depth` | estimated | max integer value of the images (256, 4096, 16384, 65536) |
| `max_iterations` / `--max-iterations` | 500 | L-BFGS iterations |

The zero-light surface is the least well determined part of the model. If you can measure it (dark
frames, or images with the shutter closed), pass `z_limits`.

## How it relates to the MATLAB version

The pipeline is a line-by-line port: the same working resolution (about 9400 pixels), entropy check
and scale-space resampling for small collections, per-pixel quantile sorting and compression to 200
quantiles, robust estimate of the intensity distribution, pivot shift, least-squares then robust
(Cauchy) regression with multi-scale Laplacian-of-Gaussian smoothness and zero-light terms. MATLAB's
`imresize`, `fspecial('log')`, `imfilter`, `hist` and `round` are reproduced exactly, and the
pre-processed data match the original code to machine precision.

The optimisation differs on purpose:

* **No minFunc.** The MATLAB code bundled minFunc, which carries a non-commercial licence. This
  version uses SciPy's L-BFGS-B, so the whole package is GPL-2.0-or-later with no non-commercial
  restriction.
* **Exact gradients.** The original gradient of the smoothness term did not match its energy (it was
  larger by the number of LoG scales), which made minFunc stop after a few dozen function
  evaluations. Here the energy is defined to match the gradient MATLAB actually followed, so the
  default parameters keep their tuned meaning, and it is minimised to convergence with an exact
  gradient.
* The first, least-squares stage is solved in closed form, and the barrier that kept the zero-light
  point in range is replaced by bound constraints.

Because the MATLAB optimiser stopped early, results are close but not identical. On synthetic
collections with known illumination (`validation/`), compared with the original MATLAB code:

| dataset | v, Python vs MATLAB (mean abs. diff.) | v error vs truth, MATLAB | v error vs truth, Python |
|---|---|---|---|
| 200 ideal images | 0.026 | 0.018 | 0.016 |
| 120 images, sparse cells | 0.005 | 0.074 | 0.071 |
| 25 images, scale-space resampling | 0.026 | 0.093 | 0.092 |

(gain surfaces normalised to mean 1). See [`validation/README.md`](validation/README.md) to repeat
the comparison with MATLAB or Octave.

The MATLAB help text listed `direct` as the default correction mode, but the code (and the ImageJ
plugin) used `zero_light_preserved`; this version follows the code.

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

GPL-2.0-or-later, see [LICENSE](LICENSE). Based on the MATLAB implementation, copyright 2015 Kevin Smith and Peter Horvath, ETH Zurich.
