import numpy as np
import pytest

from cv_fuzzy.fuzzy_engine import analyze_image_fuzzy


VALID_STATES = {"ACCEPT", "BORDERLINE", "REJECT"}
VALID_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_noise",
    "none",
}


def test_returns_contract_fields():
    image = np.random.default_rng(42).integers(
        0, 256, size=(128, 128), dtype=np.uint8
    )

    result = analyze_image_fuzzy(image)

    assert set(result) == {
        "metrics",
        "fuzzy_score",
        "state",
        "dominant_defect",
    }
    assert set(result["metrics"]) == {"blur", "contrast", "noise"}


def test_score_and_labels_are_in_contract_ranges():
    image = np.random.default_rng(7).integers(
        0, 256, size=(128, 128), dtype=np.uint8
    )

    result = analyze_image_fuzzy(image)

    assert isinstance(result["fuzzy_score"], int)
    assert 0 <= result["fuzzy_score"] <= 100
    assert result["state"] in VALID_STATES
    assert result["dominant_defect"] in VALID_DEFECTS


def test_rejects_color_images():
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D grayscale"):
        analyze_image_fuzzy(image)


def test_rejects_empty_image():
    image = np.empty((0, 0), dtype=np.uint8)

    with pytest.raises(ValueError, match="must not be empty"):
        analyze_image_fuzzy(image)


def test_rejects_non_numpy_input():
    with pytest.raises(TypeError, match="NumPy array"):
        analyze_image_fuzzy([[0, 1], [2, 3]])


def test_constant_image_does_not_crash():
    image = np.full((64, 64), 120, dtype=np.uint8)

    result = analyze_image_fuzzy(image)

    assert 0 <= result["fuzzy_score"] <= 100
    assert result["state"] in VALID_STATES
