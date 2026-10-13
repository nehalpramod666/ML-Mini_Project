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
    state_for_score,
    validate_v2_payload,
)


def _fake_m2_result(image, score=80):
    x = image.astype(np.float64)
    mean = float(x.mean())
    std = float(x.std())
    snr = math.inf if std == 0 and mean > 0 else (mean / std if std else math.inf)
    # Use the actual definitions for a test input; fuzzy score/state are independently supplied.
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
    assert payload["metrics"]["noise"] == 0.0
    encoded = json.dumps(payload, allow_nan=False)
    assert "Infinity" not in encoded and "NaN" not in encoded


def test_zero_over_zero_is_undefined_not_positive_infinity():
    image = np.zeros((16, 16), dtype=np.uint8)
    result = _fake_m2_result(image, 80)
    with pytest.raises(ContractV2Error, match="UNDEFINED_SNR"):
        build_v2_payload(result, image)


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
        "metrics": {"blur": blur_severity(5.0), "contrast": 100.0, "noise": 80.0},
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


def test_state_mismatch_and_high_contrast_gate_are_rejected():
    payload = _valid_payload()
    payload["state"] = "ACCEPT"
    with pytest.raises(ContractV2Error, match="disagrees"):
        validate_v2_payload(payload)

    payload = _valid_payload()
    payload["metrics_raw"]["contrast"] = 80.0
    payload["metrics"]["contrast"] = 100.0
    payload["dominant_defect"] = "high_contrast"
    with pytest.raises(ContractV2Error, match="disabled"):
        validate_v2_payload(payload)


def test_all_zero_severity_non_accept_is_rejected():
    payload = _valid_payload()
    payload["metrics_raw"] = {"blur": 110.0, "contrast": 50.0, "noise": 10.0}
    payload["metrics"] = {"blur": 0.0, "contrast": 0.0, "noise": 0.0}
    payload["fuzzy_score"] = 44
    payload["state"] = "REJECT"
    payload["dominant_defect"] = "motion_blur"
    with pytest.raises(ContractV2Error, match="all-zero"):
        validate_v2_payload(payload)


def test_none_requires_accept_and_zero_severities():
    payload = _valid_payload()
    payload["metrics_raw"] = {"blur": 110.0, "contrast": 50.0, "noise": 10.0}
    payload["metrics"] = {"blur": 0.0, "contrast": 0.0, "noise": 0.0}
    payload["fuzzy_score"] = 80
    payload["state"] = "ACCEPT"
    payload["dominant_defect"] = "none"
    validate_v2_payload(payload)
    payload["metrics_raw"]["noise"] = 8.0
    payload["metrics"]["noise"] = 20.0
    with pytest.raises(ContractV2Error, match="none is invalid"):
        validate_v2_payload(payload)
