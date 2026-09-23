import cv2
import numpy as np
import pytest

from cv_fuzzy.metrics import compute_blur


def _make_sharp_image() -> np.ndarray:
    """A checkerboard pattern has strong, sharp edges."""
    img = np.zeros((200, 200), dtype=np.uint8)
    img[::20, :] = 255
    img[:, ::20] = 255
    return img


def _make_blurred_version(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (15, 15), sigmaX=5)


def test_blur_score_is_float():
    img = _make_sharp_image()
    score = compute_blur(img)
    assert isinstance(score, float)


def test_sharp_image_scores_higher_than_blurred():
    sharp = _make_sharp_image()
    blurred = _make_blurred_version(sharp)
    assert compute_blur(sharp) > compute_blur(blurred)


def test_flat_image_has_near_zero_blur_score():
    flat = np.full((100, 100), 128, dtype=np.uint8)
    assert compute_blur(flat) == pytest.approx(0.0, abs=1e-6)


def test_raises_on_non_2d_input():
    bad = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_blur(bad)


def test_blur_score_on_real_sample_dicom():
    from cv_fuzzy.dicom_loader import load_dicom
    from pathlib import Path
    sample = Path(__file__).resolve().parent.parent / "sample_data" / "CT_small.dcm"
    img = load_dicom(str(sample))
    score = compute_blur(img)
    assert score >= 0