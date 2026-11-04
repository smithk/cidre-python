import numpy as np
import pytest
import tifffile

from cidre import CidreModel
from cidre.cli import main
from cidre.io import list_images, read_image

from synthetic import make_collection


@pytest.fixture(scope="module")
def image_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("images")
    imgs, _, _, _ = make_collection(n=40, shape=(64, 80), seed=5)
    for k, im in enumerate(imgs):
        tifffile.imwrite(folder / f"img_{k:03d}_w1.tif", im)
    (folder / "notes.txt").write_text("not an image")
    return folder


def test_list_images_folder_and_glob(image_folder):
    assert len(list_images(image_folder)) == 40
    assert len(list_images(str(image_folder / "img_00*_w1.tif"))) == 10
    with pytest.raises(FileNotFoundError):
        list_images(image_folder / "missing")


def test_run_writes_model_and_corrected_images(image_folder, tmp_path):
    out = tmp_path / "out"
    assert main(["-q", "run", str(image_folder), "-o", str(out), "--max-iterations", "60"]) == 0
    files = sorted(out.glob("*.tif"))
    assert len(files) == 40
    img = read_image(files[0])
    assert img.dtype == np.uint16 and img.shape == (64, 80)
    model = CidreModel.load(out / "cidre_model.mat")
    assert model.v.shape == (64, 80)


def test_fit_then_correct(image_folder, tmp_path):
    model_path = tmp_path / "w1.npz"
    assert main(["-q", "fit", str(image_folder / "*_w1.tif"), "-o", str(model_path), "--max-iterations", "60"]) == 0
    assert model_path.exists()
    out = tmp_path / "corrected"
    assert main(["-q", "correct", str(image_folder), "--model", str(model_path), "-o", str(out), "-m", "direct"]) == 0
    assert len(list(out.glob("*.tif"))) == 40


def test_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["-q", "fit", str(tmp_path / "nothing*.tif"), "-o", str(tmp_path / "m.npz")]) == 1
