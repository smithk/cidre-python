import numpy as np
import pytest
from scipy.ndimage import correlate

from cidre._matlab import fspecial_log, hist_counts, imresize, mround
from cidre.objective import filter_operator, log_sigmas


def test_mround_rounds_halves_away_from_zero():
    assert mround(0.5) == 1
    assert mround(2.5) == 3
    assert mround(-2.5) == -3
    np.testing.assert_array_equal(mround([1.49, 1.5, -0.5]), [1, 2, -1])


@pytest.mark.parametrize("method", ["bicubic", "bilinear"])
@pytest.mark.parametrize("size", [(7, 9), (40, 33), (101, 3)])
def test_imresize_preserves_constants(method, size):
    img = np.full((23, 31), 7.25)
    np.testing.assert_allclose(imresize(img, size, method=method), 7.25, rtol=0, atol=1e-12)


def test_imresize_identity_and_stack_axes():
    rng = np.random.default_rng(0)
    img = rng.random((12, 17))
    np.testing.assert_allclose(imresize(img, (12, 17)), img, atol=1e-12)
    stack = rng.random((3, 20, 30))
    out = imresize(stack, (10, 11))
    assert out.shape == (3, 10, 11)
    np.testing.assert_allclose(out[1], imresize(stack[1], (10, 11)))


def test_imresize_scale_gives_ceil_size():
    assert imresize(np.ones((7, 9)), scale=0.5).shape == (4, 5)


def test_imresize_upsampling_reproduces_ramps_in_the_interior():
    ramp = np.tile(np.arange(20.0), (10, 1))
    up = imresize(ramp, (10, 40), method="bilinear")
    # centres of output pixel j map to input coordinate (j + 0.5) / 2 - 0.5
    expected = (np.arange(40) + 0.5) / 2 - 0.5
    np.testing.assert_allclose(up[5, 2:-2], expected[2:-2], atol=1e-12)


def test_fspecial_log_is_zero_sum_and_symmetric():
    for size, sigma in [(7, 0.5), (7, 1.0), (13, 2.0)]:
        h = fspecial_log(size, sigma)
        assert abs(h.sum()) < 1e-14
        np.testing.assert_allclose(h, h.T)
        np.testing.assert_allclose(h, h[::-1, ::-1])
        assert h[size // 2, size // 2] < 0


def test_hist_counts_matches_matlab_right_closed_bins():
    np.testing.assert_array_equal(hist_counts([0.2, 1.5, 2.49, 2.5, 3.7, 9], 4), [2, 2, 0, 2])


@pytest.mark.parametrize("shape", [(9, 9), (20, 13), (5, 30)])
def test_filter_operator_matches_symmetric_imfilter(shape):
    rng = np.random.default_rng(1)
    v = rng.random(shape)
    for k, s in [(7, 1.0), (13, 2.0)]:
        h = fspecial_log(k, s)
        L = filter_operator(h, shape)
        # scipy's 'reflect' mode is MATLAB's 'symmetric' padding (also for kernels larger than the image)
        np.testing.assert_allclose((L @ v.ravel()).reshape(shape), correlate(v, h, mode="reflect"), atol=1e-12)


def test_log_sigmas_follow_the_matlab_colon_rule():
    np.testing.assert_array_equal(log_sigmas((84, 112)), [0.5, 1, 2])
    np.testing.assert_array_equal(log_sigmas((40, 30)), [0.5, 1, 2])
    np.testing.assert_array_equal(log_sigmas((300, 200)), [0.5, 1, 2, 4])  # log2(6) = 2.58 -> up to 2**2
