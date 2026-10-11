import cv2
import numpy as np
import pytest

from cv_fuzzy.metrics import compute_blur, compute_contrast, compute_noise , analyze_image

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
def test_contrast_score_is_float():
    img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    score = compute_contrast(img)

    assert isinstance(score, float)


def test_flat_image_has_zero_contrast():
    flat = np.full((100, 100), 128, dtype=np.uint8)

    assert compute_contrast(flat) == pytest.approx(0.0, abs=1e-6)


def test_high_contrast_image_scores_higher_than_low_contrast():
    # Half black, half white.
    high_contrast = np.zeros((100, 100), dtype=np.uint8)
    high_contrast[:, 50:] = 255

    # Narrow range around gray.
    low_contrast = np.random.randint(
        120, 136, (100, 100), dtype=np.uint8
    )

    assert compute_contrast(high_contrast) > compute_contrast(low_contrast)


def test_contrast_raises_on_non_2d_input():
    bad = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        compute_contrast(bad)


def test_contrast_score_on_real_sample_dicom():
    from pathlib import Path
    from cv_fuzzy.dicom_loader import load_dicom

    sample = (
        Path(__file__).resolve().parent
        / "sample_data"
        / "CT_small.dcm"
    )

    img = load_dicom(str(sample))
    score = compute_contrast(img)

    assert score >= 0
def test_noise_score_is_float():
    img = np.random.randint(80, 180, (100, 100), dtype=np.uint8)
    score = compute_noise(img)

    assert isinstance(score, float)


def test_flat_image_has_infinite_snr():
    flat = np.full((100, 100), 128, dtype=np.uint8)

    assert compute_noise(flat) == float("inf")


def test_noisier_image_has_lower_snr():
    clean = np.full((100, 100), 128, dtype=np.uint8)

    # Small variation around 128.
    low_noise = np.tile(
        np.array([126, 127, 128, 129, 130], dtype=np.uint8),
        (100, 20),
    )

    # Larger variation around the same approximate signal level.
    high_noise = np.tile(
        np.array([80, 100, 128, 156, 176], dtype=np.uint8),
        (100, 20),
    )

    assert compute_noise(low_noise) > compute_noise(high_noise)


def test_noise_raises_on_non_2d_input():
    bad = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        compute_noise(bad)


def test_noise_score_on_real_sample_dicom():
    from pathlib import Path
    from cv_fuzzy.dicom_loader import load_dicom

    sample = (
        Path(__file__).resolve().parent
        / "sample_data"
        / "CT_small.dcm"
    )

    img = load_dicom(str(sample))
    score = compute_noise(img)

    assert isinstance(score, float)
    assert score >= 0
def test_analyze_image_returns_expected_structure():
    img = np.zeros((64, 64), dtype=np.uint8)

    result = analyze_image(img)

    assert set(result.keys()) == {"metrics"}
    assert set(result["metrics"].keys()) == {
        "blur",
        "contrast",
        "noise",
    }
# Multi-source blur calibration robustness tests

def test_blur_score_for_2d_gradient_is_finite():
    """A smooth 2D gradient should produce a valid finite score."""
    y, x = np.indices((128, 128))
    gradient = ((x + y) / 2).astype(np.uint8)

    score = compute_blur(gradient)

    assert np.isfinite(score)
    assert score >= 0


def test_blur_score_decreases_with_stronger_blur_on_sharp_edge():
    """Increasing blur strength should reduce an edge-rich image's score."""
    sharp = np.zeros((128, 128), dtype=np.uint8)
    sharp[:, 64:] = 255

    settings = [
        (3, 0.5),
        (5, 1.0),
        (9, 2.0),
        (15, 5.0),
    ]

    scores = [
        compute_blur(
            cv2.GaussianBlur(sharp, (kernel, kernel), sigmaX=sigma)
        )
        for kernel, sigma in settings
    ]

    assert all(
        earlier > later
        for earlier, later in zip(scores, scores[1:])
    )


def test_blur_score_for_flat_image_is_finite_and_near_zero():
    """A flat image should not produce NaN or infinity."""
    flat = np.full((128, 128), 128, dtype=np.uint8)

    score = compute_blur(flat)

    assert np.isfinite(score)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_blur_score_is_reproducible():
    """Identical input and processing should produce identical scores."""
    sharp = _make_sharp_image()
    blurred = cv2.GaussianBlur(sharp, (5, 5), sigmaX=1.0)

    first_score = compute_blur(blurred)
    second_score = compute_blur(blurred.copy())

    assert first_score == second_score
