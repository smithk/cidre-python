# Validation against the original MATLAB code

These scripts run the original MATLAB implementation (`trunk/matlab` in
[smithk/cidre](https://github.com/smithk/cidre)) and this Python port on the same synthetic image
collections, whose true gain `v` and zero-light surface `z` are known.

```bash
git clone https://github.com/smithk/cidre ../cidre   # the original code, next to this repository
cd validation
python make_data.py                      # writes data/{iid,cells,few}.{npz,mat}

# original code, in MATLAB or GNU Octave (with the image package)
octave-cli --eval "run_reference('../../cidre/trunk/matlab', 'data/cells.mat', 'data/cells_ref.mat')"

python compare.py cells                  # or no argument for all datasets
```

In Octave, `octave_shims/` provides MATLAB-exact `imresize` (bicubic with antialiasing) and
`fspecial('log')` (zero-sum), which differ in Octave's image package, and minFunc runs without its
MATLAB-only mex files. In MATLAB nothing is shadowed.

## Results (original code run in Octave 8.4)

| dataset | pre-processed stack, max rel. diff. | v, Python vs MATLAB, mean / max abs. diff. | v error vs truth, MATLAB / Python | z mean: truth / MATLAB / Python |
|---|---|---|---|---|
| `iid`: 200 ideal images | 9e-16 | 0.026 / 0.128 | 0.018 / 0.016 | 100 / 148 / 30 |
| `cells`: 120 images, sparse cells | 5e-16 | 0.005 / 0.028 | 0.074 / 0.071 | 100 / 111 / 101 |
| `few`: 25 images, scale-space path | 3e-16 | 0.026 / 0.106 | 0.093 / 0.092 | 100 / 116 / 69 |

Gain surfaces are compared after normalising to mean 1, at the working resolution.

What this shows:

* Everything up to the optimisation (resizing, entropy, scale-space resampling, sorting,
  compression, the estimate of `Q`, the pivot shift and the least-squares stage) agrees with the
  MATLAB code to machine precision.
* The robust optimisation reaches a lower energy than the MATLAB run, which stopped after 34 to 80
  function evaluations. The gain surfaces agree to 0.5 to 2.6 % on average and are at least as close
  to the truth.
* The zero-light surface is weakly determined by the data in both implementations. When it matters,
  measure it and pass `z_limits`.

### Why the MATLAB optimisation stops early

`cdr_objective.m` computes the smoothness energy as `sum((imfilter(v, h) / K).^2)` over `K` LoG
scales, but its gradient as `imfilter(2 * imfilter(v, h) / K, h)`, which is `K` times the gradient of
that energy. With function values and gradients that disagree, minFunc's line search stalls and the
optimiser exits long before `max_lbfgs_iterations` (on `few`: "Step Size below progTol" after 72
evaluations). Dividing that gradient by `K` in `cdr_objective.m` makes the same run continue to the
500-iteration limit. The Python port defines the
smoothness energy as `sum(imfilter(v, h).^2) / K`, the energy whose gradient MATLAB was following,
so that the tuned defaults (`lambda_v = 6` and the rule for few images) keep their meaning, and
minimises it to convergence with an exact gradient. Using the literal `1 / K^2` energy instead gives
gain surfaces that are noticeably less smooth than MATLAB's and further from the truth on the
`cells` and `few` datasets.
