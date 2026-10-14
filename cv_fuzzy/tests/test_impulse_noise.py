
import cv2
import numpy as np
import pytest

from cv_fuzzy.impulse_noise import estimate_impulse_noise
from cv_fuzzy.synthetic_degrade import reduce_contrast


def make_midrange_image():
    """Create deterministic image structure without saturated pixels."""
    y, x = np.mgrid[:96, :96]
    values = (
        110
        + 0.5 * x
        + 0.25 * y
        + 15 * np.sin(x / 7.0)
    )
    return np.clip(values, 40, 210).astype(np.uint8)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("all_black", 0),
        ("all_white", 255),
        ("constant", 128),
    ],
)
def test_constant_images_have_no_detected_impulses(name, value):
    image = np.full((64, 64), value, dtype=np.uint8)

    result = estimate_impulse_noise(image)

    assert result["impulse_pixels"] == 0
    assert result["impulse_fraction"] == 0.0


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_pixels(bad_value):
    image = np.full((16, 16), 128.0)
    image[0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        estimate_impulse_noise(image)


def test_detects_isolated_black_and_white_impulses():
    image = np.full((128, 128), 128, dtype=np.uint8)
    image[::10, ::10] = 0
    image[5::10, 5::10] = 255

    result = estimate_impulse_noise(image)

    assert result["impulse_pixels"] > 0
    assert result["impulse_fraction"] > 0
    assert result["impulse_percent"] > 0


def test_blurred_midrange_image_does_not_trigger_impulse_detection():
    image = make_midrange_image()
    blurred = cv2.GaussianBlur(image, (9, 9), 2.0)

    result = estimate_impulse_noise(blurred)

    assert result["impulse_pixels"] == 0
    assert result["impulse_fraction"] == 0.0


def test_reduced_contrast_image_does_not_trigger_impulse_detection():
    image = make_midrange_image()
    reduced = reduce_contrast(image, factor=0.3)

    result = estimate_impulse_noise(reduced)

    assert result["impulse_pixels"] == 0
    assert result["impulse_fraction"] == 0.0


def test_rejects_color_images():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D"):
        estimate_impulse_noise(image)



def test_rejects_even_median_kernel_size():
    image = np.full((32, 32), 128, dtype=np.uint8)

    with pytest.raises(ValueError, match="odd integer"):
        estimate_impulse_noise(image, kernel_size=2)
