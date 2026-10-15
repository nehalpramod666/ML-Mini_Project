import json
import math

import numpy as np
import pytest

from cv_fuzzy.v2_contract import (
    ContractV2Error,
    blur_severity,
    build_v2_payload,
    contrast_severity_parts,
    noise_severity,
    round_half_even,
    state_for_score,
    validate_v2_payload,
)


def _fake_m2_result(image, score=80):
    x = image.astype(np.float64)
    mean = float(x.mean())
    std = float(x.std())
    snr = math.inf if std == 0 and mean > 0 else (mean / std if std else math.inf)
    import cv2
    blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
    contrast = std
    return {
        "metrics": {"blur": blur, "contrast": contrast, "noise": snr},
        "fuzzy_score": score,
        "state": state_for_score(score),
        "dominant_defect": "none",
    }


def test_blur_exact_anchors_and_monotonicity():
    assert blur_severity(0) == pytest.approx(100.0)
    assert blur_severity(110) == pytest.approx(0.0)
    assert blur_severity(200) == pytest.approx(0.0)
    values = [blur_severity(v) for v in [0, 1, 5, 20, 50, 80, 110, 1000]]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_noise_anchors_clipping_and_monotonicity():
    expected = {0: 100, 1.5: 85, 2.5: 75, 4: 60, 7: 30, 10: 0, 20: 0}
    for raw, severity in expected.items():
        assert noise_severity(raw) == pytest.approx(severity)
    values = [noise_severity(v) for v in [0, 1, 1.5, 2.5, 4, 7, 10, 100]]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert noise_severity(math.inf) == 0
    with pytest.raises(ContractV2Error):
        noise_severity(math.nan)
    with pytest.raises(ContractV2Error):
        noise_severity(-math.inf)


def test_contrast_ramp_endpoints_and_optimal_band():
    assert contrast_severity_parts(20) == pytest.approx((100, 100, 0))
    assert contrast_severity_parts(29) == pytest.approx((50, 50, 0))
    assert contrast_severity_parts(38) == pytest.approx((0, 0, 0))
    assert contrast_severity_parts(50) == pytest.approx((0, 0, 0))
    assert contrast_severity_parts(60) == pytest.approx((0, 0, 0))
    assert contrast_severity_parts(70) == pytest.approx((50, 0, 50))
    assert contrast_severity_parts(80) == pytest.approx((100, 0, 100))


def test_round_half_even_ties_and_clamping():
    # Exact half values round to nearest even integer
    assert round_half_even(22.5) == 22
    assert round_half_even(23.5) == 24
    assert round_half_even(2.5) == 2
    assert round_half_even(3.5) == 4
    assert round_half_even(0.5) == 0
    assert round_half_even(1.5) == 2
    # Clamping tests
    assert round_half_even(-10.0) == 0
    assert round_half_even(115.0) == 100
    assert round_half_even(0.0) == 0
    assert round_half_even(100.0) == 100


def test_state_boundaries():
    assert state_for_score(44) == "REJECT"
    assert state_for_score(45) == "BORDERLINE"
    assert state_for_score(74) == "BORDERLINE"
    assert state_for_score(75) == "ACCEPT"
    with pytest.raises(ContractV2Error):
        state_for_score(75.0)
    with pytest.raises(ContractV2Error):
        state_for_score(True)


def test_constant_positive_image_encodes_infinite_snr_as_null():
    image = np.full((32, 32), 100, dtype=np.uint8)
    result = _fake_m2_result(image, 80)
    payload = build_v2_payload(result, image)
    assert payload["metrics_raw"]["noise"] is None
    assert payload["metrics_raw_status"]["noise"] == "positive_infinity"
    assert payload["metrics"]["noise"] == 0
    assert isinstance(payload["metrics"]["noise"], int)
    encoded = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded


def test_zero_over_zero_is_undefined_not_positive_infinity():
    image = np.zeros((16, 16), dtype=np.uint8)
    result = _fake_m2_result(image, 80)
    with pytest.raises(ContractV2Error) as exc_info:
        build_v2_payload(result, image)
    assert exc_info.value.code == "UNDEFINED_SNR"
    assert "UNDEFINED_SNR" in str(exc_info.value)


def test_raw_nan_and_negative_infinity_are_rejected():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = _fake_m2_result(image, 80)
    result["metrics"]["blur"] = math.nan
    with pytest.raises(ContractV2Error):
        build_v2_payload(result, image)

    result = _fake_m2_result(image, 80)
    result["metrics"]["noise"] = -math.inf
    with pytest.raises(ContractV2Error):
        build_v2_payload(result, image)


def test_mismatched_source_snr_is_rejected():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    result = _fake_m2_result(image, 80)
    result["metrics"]["noise"] *= 2
    with pytest.raises(ContractV2Error, match="does not match"):
        build_v2_payload(result, image)


def _valid_payload():
    return {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {"blur": 5.0, "contrast": 10.0, "noise": 2.0},
        "metrics_raw_status": {"blur": "finite", "contrast": "finite", "noise": "finite"},
        "metrics": {"blur": round_half_even(blur_severity(5.0)), "contrast": 100, "noise": 80},
        "fuzzy_score": 60,
        "state": "BORDERLINE",
        "dominant_defect": "low_contrast",
    }


def test_status_value_relationship_is_strict():
    payload = _valid_payload()
    payload["metrics_raw"]["noise"] = None
    payload["metrics_raw_status"]["noise"] = "finite"
    with pytest.raises(ContractV2Error):
        validate_v2_payload(payload)

    payload = _valid_payload()
    payload["metrics_raw"]["noise"] = 2.0
    payload["metrics_raw_status"]["noise"] = "positive_infinity"
    with pytest.raises(ContractV2Error):
        validate_v2_payload(payload)


def test_metrics_must_be_integers_only():
    payload = _valid_payload()
    # Fractional float must be rejected
    payload["metrics"]["blur"] = 62.0
    with pytest.raises(ContractV2Error, match="must be an integer"):
        validate_v2_payload(payload)

    # Boolean must be rejected
    payload["metrics"]["blur"] = True
    with pytest.raises(ContractV2Error, match="must be an integer"):
        validate_v2_payload(payload)

    # Out-of-range integer must be rejected
    payload["metrics"]["blur"] = 101
    with pytest.raises(ContractV2Error, match="within"):
        validate_v2_payload(payload)


def test_state_mismatch_and_high_contrast_gate_are_rejected():
    payload = _valid_payload()
    payload["state"] = "ACCEPT"
    with pytest.raises(ContractV2Error, match="disagrees"):
        validate_v2_payload(payload)

    payload = _valid_payload()
    payload["metrics_raw"]["contrast"] = 80.0
    payload["metrics"]["contrast"] = 100
    payload["dominant_defect"] = "high_contrast"
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "DISABLED_HIGH_CONTRAST_TRIGGER"


def test_all_zero_severity_non_accept_is_rejected():
    payload = _valid_payload()
    payload["metrics_raw"] = {"blur": 110.0, "contrast": 50.0, "noise": 10.0}
    payload["metrics"] = {"blur": 0, "contrast": 0, "noise": 0}
    payload["fuzzy_score"] = 44
    payload["state"] = "REJECT"
    payload["dominant_defect"] = "motion_blur"
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "CONSISTENCY_ERROR"


def test_none_requires_accept_and_zero_severities():
    payload = _valid_payload()
    payload["metrics_raw"] = {"blur": 110.0, "contrast": 50.0, "noise": 10.0}
    payload["metrics"] = {"blur": 0, "contrast": 0, "noise": 0}
    payload["fuzzy_score"] = 80
    payload["state"] = "ACCEPT"
    payload["dominant_defect"] = "none"
    validate_v2_payload(payload)

    payload["metrics_raw"]["noise"] = 8.0
    payload["metrics"]["noise"] = 20
    with pytest.raises(ContractV2Error, match="none is invalid") as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "CONSISTENCY_ERROR"


def test_dominant_defect_tie_priority():
    # Tie priority is motion_blur > low_contrast > low_snr
    # Case 1: motion_blur ties low_contrast (e.g. blur=5.0 -> 62, contrast=26.84 -> 62)
    # Using raw values that yield exact integer severities:
    # contrast 20.0 -> low_contrast 100
    # blur 0.0 -> motion_blur 100
    # noise 0.0 -> low_snr 100
    payload = {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {"blur": 0.0, "contrast": 20.0, "noise": 0.0},
        "metrics_raw_status": {"blur": "finite", "contrast": "finite", "noise": "finite"},
        "metrics": {"blur": 100, "contrast": 100, "noise": 100},
        "fuzzy_score": 20,
        "state": "REJECT",
        "dominant_defect": "motion_blur",  # motion_blur beats low_contrast and low_snr
    }
    validate_v2_payload(payload)

    # If set to low_contrast or low_snr when tied with motion_blur, must raise
    payload["dominant_defect"] = "low_contrast"
    with pytest.raises(ContractV2Error, match="tie-break"):
        validate_v2_payload(payload)

    # Case 2: low_contrast ties low_snr when motion_blur is lower
    payload["metrics_raw"]["blur"] = 50.0
    payload["metrics"]["blur"] = round_half_even(blur_severity(50.0))  # ~16
    payload["dominant_defect"] = "low_contrast"  # low_contrast beats low_snr
    validate_v2_payload(payload)

    payload["dominant_defect"] = "low_snr"
    with pytest.raises(ContractV2Error, match="tie-break"):
        validate_v2_payload(payload)


def test_disabled_high_contrast_tie_and_win_rejected():
    # If high contrast ties or wins while trigger is disabled, raise DISABLED_HIGH_CONTRAST_TRIGGER
    # contrast 80.0 -> high_contrast 100
    # blur 20.0 -> blur_severity(20.0) ~ 35
    # noise 7.0 -> noise_severity(7.0) = 30
    payload = {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {"blur": 20.0, "contrast": 80.0, "noise": 7.0},
        "metrics_raw_status": {"blur": "finite", "contrast": "finite", "noise": "finite"},
        "metrics": {
            "blur": round_half_even(blur_severity(20.0)),
            "contrast": 100,
            "noise": 30,
        },
        "fuzzy_score": 40,
        "state": "REJECT",
        "dominant_defect": "high_contrast",
    }
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload, allow_high_contrast=False)
    assert exc_info.value.code == "DISABLED_HIGH_CONTRAST_TRIGGER"

    # Even if caller attempts to assign enabled label when high_contrast won:
    payload["dominant_defect"] = "motion_blur"
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload, allow_high_contrast=False)
    assert exc_info.value.code == "DISABLED_HIGH_CONTRAST_TRIGGER"


def test_disabled_high_contrast_subordinate_positive_allowed():
    # High contrast is positive but strictly lower than the enabled winner
    # contrast 66.0 -> high_contrast = 100 * (66 - 60) / 20 = 30.0 -> 30
    # blur 0.0 -> blur_s = 100 -> 100 (enabled winner)
    # noise 10.0 -> 0
    payload = {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {"blur": 0.0, "contrast": 66.0, "noise": 10.0},
        "metrics_raw_status": {"blur": "finite", "contrast": "finite", "noise": "finite"},
        "metrics": {
            "blur": 100,
            "contrast": 30,  # reflects max(low=0, high=30)
            "noise": 0,
        },
        "fuzzy_score": 30,
        "state": "REJECT",
        "dominant_defect": "motion_blur",  # enabled winner exceeds high_contrast (100 > 30)
    }
    # This must be allowed without error
    validate_v2_payload(payload, allow_high_contrast=False)


def test_rejection_of_legacy_and_internal_noise_labels():
    payload = _valid_payload()
    payload["dominant_defect"] = "high_noise"
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "INVALID_SCHEMA"

    payload["dominant_defect"] = "sensor_noise"
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "INVALID_SCHEMA"


def test_state_and_dominant_defect_schema_hardening():
    payload = _valid_payload()

    # Invalid state enum string raises INVALID_SCHEMA, not CONSISTENCY_ERROR
    payload["state"] = "INVALID_STATE"
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "INVALID_SCHEMA"

    # Non-string state raises INVALID_SCHEMA
    for bad_state in (123, True, None, ["ACCEPT"]):
        payload["state"] = bad_state
        with pytest.raises(ContractV2Error) as exc_info:
            validate_v2_payload(payload)
        assert exc_info.value.code == "INVALID_SCHEMA"

    # Reset state to valid
    payload["state"] = "BORDERLINE"

    # Non-string or unhashable dominant_defect raises INVALID_SCHEMA without leaking TypeError
    for bad_defect in (123, True, None, ["motion_blur"], {"defect": "none"}):
        payload["dominant_defect"] = bad_defect
        with pytest.raises(ContractV2Error) as exc_info:
            validate_v2_payload(payload)
        assert exc_info.value.code == "INVALID_SCHEMA"

    # Genuine score/state mismatch raises CONSISTENCY_ERROR
    payload["dominant_defect"] = "low_contrast"
    payload["state"] = "ACCEPT"  # Valid enum in STATES, but disagrees with fuzzy_score 60
    with pytest.raises(ContractV2Error) as exc_info:
        validate_v2_payload(payload)
    assert exc_info.value.code == "CONSISTENCY_ERROR"
