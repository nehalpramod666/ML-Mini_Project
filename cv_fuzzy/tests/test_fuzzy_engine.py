import numpy as np
import pytest

from cv_fuzzy.fuzzy_engine import FuzzyInferenceError, analyze_image_fuzzy


VALID_STATES = {"ACCEPT", "BORDERLINE", "REJECT"}

# VALID_DEFECTS includes both the legacy "high_noise" (currently emitted by the
# fuzzy engine) and "sensor_noise" (used in v2_contract.py).  The discrepancy
# is tracked as P2/P11 and will be resolved once D7 is signed.  Do not remove
# either entry until the label vocabulary is frozen.
VALID_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_noise",   # legacy label — interim, pending D7 label decision
    "sensor_noise", # v2 label — interim, pending D7 label decision
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


# ---------------------------------------------------------------------------
# P3 regression: formerly-uncovered rule combination
# ---------------------------------------------------------------------------
# Background: the original rule base (commit 8427b0c) had no consequent for
# the combination blur=high AND contrast=high AND snr=medium.  skfuzzy raises
# KeyError when no output membership is activated.  The fix adds a coverage
# rule: blur["high"] & contrast["high"] & snr["medium"] -> quality["fair"].
#
# CALIBRATION NOTE: quality="fair" for this combination is a proposed
# conservative choice.  It has not been reviewed or approved by M2.
# Do not interpret passing tests as evidence of calibration accuracy.
# ---------------------------------------------------------------------------

# These are the exact documented gap values from the Phase 0 audit.
# Verified by direct membership evaluation:
#   blur=200 -> blur[low]=0, blur[medium]=0, blur[high]=1
#   contrast=70 -> contrast[low]=0, contrast[medium]=0, contrast[high]=1
#   snr=3.0  -> snr[low]=0, snr[medium]=0.75, snr[high]=0
# The combination is therefore blur=high AND contrast=high AND snr=medium.
_DOCUMENTED_GAP_INPUTS = [
    (200.0, 70.0, 3.0),   # exact values from Phase 0 audit report
    (150.0, 60.0, 3.5),   # pure high/high/medium (snr[medium]=1.0)
    (110.0, 65.0, 4.0),   # near membership boundaries
    (500.0, 75.0, 3.5),   # high blur well above threshold
]

@pytest.mark.parametrize("blur_in,contrast_in,snr_in", _DOCUMENTED_GAP_INPUTS)
def test_documented_gap_inputs_directly(blur_in, contrast_in, snr_in):
    """Regression test for P3: inject the exact documented gap inputs directly
    into the fuzzy control system, bypassing image construction.

    This definitively proves the coverage rule fires and produces output
    without KeyError.  It does not claim calibration accuracy.
    """
    from skfuzzy import control as ctrl
    from cv_fuzzy.fuzzy_engine import _CONTROL_SYSTEM

    sim = ctrl.ControlSystemSimulation(_CONTROL_SYSTEM)
    sim.input["blur"] = blur_in
    sim.input["contrast"] = contrast_in
    sim.input["snr"] = snr_in
    sim.compute()
    quality = sim.output["quality"]

    assert isinstance(quality, float)
    assert 0.0 <= quality <= 100.0, f"quality={quality} out of range for inputs ({blur_in},{contrast_in},{snr_in})"


def test_documented_gap_membership_profile():
    """Assert the documented gap inputs (blur=200, contrast=70, snr=3.0)
    actually produce the claimed blur=high, contrast=high, snr=medium
    fuzzy memberships.

    This test locks down the membership profile so a future change to
    membership functions that shifts the profile cannot silently pass.
    """
    import skfuzzy as _fuzz
    from cv_fuzzy.fuzzy_engine import (
        _BLUR, _CONTRAST, _SNR, BLUR_MAX, CONTRAST_MAX, SNR_MAX,
    )
    import numpy as _np

    blur_in, contrast_in, snr_in = 200.0, 70.0, 3.0

    b_high = _fuzz.interp_membership(_BLUR.universe, _BLUR["high"].mf, blur_in)
    b_low = _fuzz.interp_membership(_BLUR.universe, _BLUR["low"].mf, blur_in)
    b_med = _fuzz.interp_membership(_BLUR.universe, _BLUR["medium"].mf, blur_in)

    c_high = _fuzz.interp_membership(_np.linspace(0, CONTRAST_MAX, 401), _CONTRAST["high"].mf, contrast_in)
    c_low = _fuzz.interp_membership(_np.linspace(0, CONTRAST_MAX, 401), _CONTRAST["low"].mf, contrast_in)
    c_med = _fuzz.interp_membership(_np.linspace(0, CONTRAST_MAX, 401), _CONTRAST["medium"].mf, contrast_in)

    s_med = _fuzz.interp_membership(_np.linspace(0, SNR_MAX, 201), _SNR["medium"].mf, snr_in)
    s_low = _fuzz.interp_membership(_np.linspace(0, SNR_MAX, 201), _SNR["low"].mf, snr_in)
    s_high = _fuzz.interp_membership(_np.linspace(0, SNR_MAX, 201), _SNR["high"].mf, snr_in)

    # Confirm the documented gap profile.
    assert b_high == pytest.approx(1.0), f"Expected blur[high]=1.0, got {b_high}"
    assert b_low  == pytest.approx(0.0), f"Expected blur[low]=0.0,  got {b_low}"
    assert b_med  == pytest.approx(0.0), f"Expected blur[med]=0.0,  got {b_med}"

    assert c_high == pytest.approx(1.0), f"Expected contrast[high]=1.0, got {c_high}"
    assert c_low  == pytest.approx(0.0), f"Expected contrast[low]=0.0,  got {c_low}"
    assert c_med  == pytest.approx(0.0), f"Expected contrast[med]=0.0,  got {c_med}"

    assert s_med == pytest.approx(0.75), f"Expected snr[medium]=0.75, got {s_med}"
    assert s_low  == pytest.approx(0.0), f"Expected snr[low]=0.0,  got {s_low}"
    assert s_high == pytest.approx(0.0), f"Expected snr[high]=0.0, got {s_high}"


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
def test_random_images_do_not_raise_key_error(seed):
    """Randomly-generated images should produce valid output, not KeyError.

    FuzzyInferenceError wraps any KeyError that slips past the rule base.
    The new coverage rule should prevent this on typical random inputs.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    result = analyze_image_fuzzy(image)

    assert isinstance(result["fuzzy_score"], int)
    assert 0 <= result["fuzzy_score"] <= 100
    assert result["state"] in VALID_STATES


def test_fuzzy_inference_error_is_importable():
    """FuzzyInferenceError must be accessible so callers can catch it distinctly."""
    assert issubclass(FuzzyInferenceError, RuntimeError)


def test_analyze_image_fuzzy_raises_fuzzy_inference_error_when_quality_unavailable(monkeypatch):
    """When fuzzy inference produces no 'quality' output (e.g. uncovered rule combination),
    analyze_image_fuzzy() must raise FuzzyInferenceError rather than leaking a raw KeyError.

    Uses pytest monkeypatching to simulate a missing quality output deterministically,
    without altering production rules or depending on an uncovered combination.
    Passage of this test verifies exception-handling mechanics and does NOT imply
    calibration or clinical validity.
    """
    class FakeSimulationMissingQuality:
        def __init__(self, *args, **kwargs):
            self.input = {}
            self.output = {}  # "quality" key is omitted

        def compute(self):
            # Compute runs cleanly, but produces no "quality" output
            pass

    import cv_fuzzy.fuzzy_engine as fe
    monkeypatch.setattr(fe.ctrl, "ControlSystemSimulation", FakeSimulationMissingQuality)

    image = np.full((32, 32), 128, dtype=np.uint8)

    with pytest.raises(FuzzyInferenceError) as exc_info:
        analyze_image_fuzzy(image)

    err = exc_info.value
    err_msg = str(err)

    # Verify the exception message provides useful context about inputs and missing output
    assert "blur=" in err_msg
    assert "contrast=" in err_msg
    assert "snr=" in err_msg
    assert "quality" in err_msg
    assert "The rule base does not cover this input combination" in err_msg

    # Verify that the direct underlying cause was KeyError, not leaked raw
    assert isinstance(err.__cause__, KeyError)
    assert err.__cause__.args == ("quality",)

@pytest.mark.parametrize(
    "bad_value",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_rejects_non_finite_pixel_values(bad_value):
    image = np.zeros((32, 32), dtype=np.float64)
    image[0, 0] = bad_value

    with pytest.raises(FuzzyInferenceError, match="finite pixel values"):
        analyze_image_fuzzy(image)
