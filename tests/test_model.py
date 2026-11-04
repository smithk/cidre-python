import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import cidre
from cidre import CidreModel
from cidre.model import default_lambda_v, estimate_Q
from cidre.preprocess import compress_stack, estimate_bit_depth, working_size

from synthetic import true_surfaces


@pytest.fixture(scope="module")
def iid_collection():
    """Images whose per-pixel intensity distributions are identical up to v and z:
    exactly CIDRE's assumption, so the model must be recovered closely."""
    rng = np.random.default_rng(0)
    shape = (90, 120)
    v, z = true_surfaces(shape, z_level=100.0)
    L = rng.gamma(2.0, 400.0, (200,) + shape)
    L = np.stack([gaussian_filter(l, 2) for l in L])
    L = L / L.mean() * 800
    I = np.clip(np.round(v * L + z + 3 * rng.standard_normal(L.shape)), 0, 65535).astype(np.uint16)
    return I, L, v, z


@pytest.fixture(scope="module")
def fitted(iid_collection):
    return cidre.fit(iid_collection[0])


def test_working_size_and_bit_depth():
    assert working_size((1024, 1344)) == (85, 111)
    assert working_size((2048, 2048)) == (97, 97)
    assert estimate_bit_depth(255) == 2**8
    assert estimate_bit_depth(4000) == 2**12
    assert estimate_bit_depth(5000) == 2**14
    assert estimate_bit_depth(40000) == 2**16


def test_default_lambda_v():
    assert default_lambda_v(0) == 9.5
    assert default_lambda_v(100) == pytest.approx(7.75)
    assert default_lambda_v(200) == 6.0
    assert default_lambda_v(5000) == 6.0


def test_compress_stack_averages_consecutive_slices():
    S = np.arange(1000.0)[:, None, None] * np.ones((1, 2, 3))
    out = compress_stack(S, 200)
    assert out.shape == (200, 2, 3)
    assert np.all(np.diff(out[:, 0, 0]) > 0)
    assert out[0, 0, 0] == pytest.approx(np.mean(np.arange(0, 6)))  # slices 1..6 (1-based)
    assert compress_stack(S[:150], 200) is not None and compress_stack(S[:150], 200).shape[0] == 150


def test_estimate_Q_uses_the_central_pixels():
    S = np.sort(np.random.default_rng(0).random((10, 6, 7)), axis=0)
    gain = np.linspace(0.5, 1.5, 42).reshape(6, 7)
    Q = estimate_Q(S * gain, 1.0)
    assert Q.shape == (10,)
    assert np.all(np.diff(Q) >= 0)


def test_recovers_gain_on_ideal_data(iid_collection, fitted):
    _, _, v, _ = iid_collection
    vn, vt = fitted.v / fitted.v.mean(), v / v.mean()
    err = np.abs(vn - vt)
    assert fitted.v.shape == v.shape
    assert err.mean() < 0.03
    assert np.corrcoef(vn.ravel(), vt.ravel())[0, 1] > 0.98


def test_correction_flattens_the_illumination(iid_collection, fitted):
    I, L, _, _ = iid_collection
    corrected = fitted.correct(I, "direct", dtype=float)
    # after correction the mean image should be (close to) flat
    before = I.mean(axis=0)
    after = corrected.mean(axis=0)
    cv = lambda a: a.std() / a.mean()  # noqa: E731
    assert cv(after) < 0.25 * cv(before)


def test_correction_modes_follow_the_formulas(fitted):
    rng = np.random.default_rng(1)
    img = rng.integers(200, 3000, fitted.image_size).astype(np.uint16)
    I = img.astype(float)
    v, z = fitted.v, fitted.z
    base = (I - z) / v
    np.testing.assert_allclose(fitted.correct(img, 2, dtype=None), base)
    np.testing.assert_allclose(fitted.correct(img, "dynamic_range_corrected", dtype=None), base * v.mean())
    np.testing.assert_allclose(fitted.correct(img, "zero_light_preserved", dtype=None), base * v.mean() + z.mean())
    out = fitted.correct(img)
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, np.clip(np.floor(base * v.mean() + z.mean() + 0.5), 0, 65535))
    with pytest.raises(ValueError):
        fitted.correct(img, "nonsense")
    with pytest.raises(ValueError):
        fitted.correct(img[:10])


def test_integer_output_saturates_like_matlab():
    model = CidreModel(v=np.full((2, 2), 0.5), z=np.full((2, 2), 100.0))
    img = np.array([[0, 50], [150, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(model.correct(img, "direct"), [[0, 0], [100, 255]])


@pytest.mark.parametrize("suffix", [".npz", ".mat"])
def test_save_and_load_roundtrip(tmp_path, fitted, suffix):
    path = fitted.save(tmp_path / f"model{suffix}")
    m = CidreModel.load(path)
    np.testing.assert_allclose(m.v, fitted.v)
    np.testing.assert_allclose(m.z, fitted.z)
    np.testing.assert_allclose(m.v_small, fitted.v_small)
    assert m.info["lambda_v"] == pytest.approx(fitted.info["lambda_v"])


def test_load_fiji_csv_model(tmp_path):
    v = np.linspace(0.5, 1.5, 12).reshape(3, 4)
    z = np.full((3, 4), 99.0)
    np.savetxt(tmp_path / "cidre_model_v.csv", v, delimiter=",")
    np.savetxt(tmp_path / "cidre_model_z.csv", z, delimiter=",")
    m = CidreModel.load(tmp_path)
    np.testing.assert_allclose(m.v, v)
    np.testing.assert_allclose(m.z, z)


def test_z_limits_are_respected(iid_collection):
    m = cidre.fit(iid_collection[0][:60], z_limits=(95, 105), max_iterations=100)
    zx, zy = m.info["zero_light_point"]
    assert 95 <= zx <= 105 and 95 <= zy <= 105
    assert m.info["z_limits"] == [95, 105]


def test_few_images_trigger_scale_space_resampling(iid_collection, caplog):
    with caplog.at_level("WARNING"):
        m = cidre.fit(iid_collection[0][:8, :, :], max_iterations=50)
    assert "scale-space" in caplog.text
    assert m.info["num_images"] == 8
    assert m.info["num_quantiles"] > 8
