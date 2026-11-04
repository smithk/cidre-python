import numpy as np

from cidre.objective import CidreProblem


def _problem(seed=0, Z=30, R=12, C=15):
    rng = np.random.default_rng(seed)
    S = np.sort(rng.gamma(3, 100, (Z, R, C)), axis=0) + 50
    Q = S[:, R // 2, C // 2].copy()
    mid = Z // 2
    p = CidreProblem(S - S[mid], Q - Q[mid], float(Q[mid]), S[mid].copy(), 6.0, 0.5, 40.0)
    x = p.pack(1 + 0.1 * rng.standard_normal((R, C)), 10 * rng.standard_normal((R, C)), 30.0, 45.0)
    return p, x, rng


def test_gradient_matches_finite_differences():
    p, x, rng = _problem()
    _, g = p(x)
    idx = list(rng.choice(x.size - 2, 25, replace=False)) + [x.size - 2, x.size - 1]
    for i in idx:
        h = 1e-5 * max(1.0, abs(x[i]))
        e = np.zeros_like(x)
        e[i] = h
        num = (p(x + e)[0] - p(x - e)[0]) / (2 * h)
        assert abs(g[i] - num) <= 1e-4 * max(10.0, abs(num)), (i, g[i], num)


def test_energy_is_the_weighted_sum_of_its_terms():
    p, x, _ = _problem(seed=3)
    E, _ = p(x)
    t = p.energy_terms(x)
    expected = t["fit"] + 10**p.lambda_v * t["vreg"] + 10**p.lambda_z * t["zero"]
    assert np.isclose(E, expected, rtol=1e-12)


def test_smoothness_term_vanishes_for_constant_gain():
    p, x, _ = _problem()
    v = np.full(p.shape, 1.3)
    assert abs(v.ravel() @ (p.A @ v.ravel())) < 1e-12
