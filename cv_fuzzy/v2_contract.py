"""Provisional M2 contract-v2 calibration/validation helpers.

Do not call this calibration clinically validated. The numeric anchors are draft values
for review. A successful result is returned only if raw statuses, score/state, labels,
and strict JSON compatibility are internally consistent.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

CONTRACT_VERSION = 2
METRICS_SCALE = "severity_0_100"
RAW_KEYS = ("blur", "contrast", "noise")
LABELS = {"motion_blur", "low_contrast", "high_contrast", "sensor_noise", "none"}
STATES = {"ACCEPT", "BORDERLINE", "REJECT"}
HIGH_CONTRAST_TRIGGER_ENABLED = False


class ContractV2Error(ValueError):
    """Raised when a v2 assessment must not be serialized as a successful result."""


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractV2Error(f"{field} must be a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ContractV2Error(f"{field} must be a finite non-negative number")
    return value


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(np.clip(value, lower, upper))


def blur_severity(variance: float) -> float:
    """Draft formula; the variance cap 110 is provisional."""
    v = _finite_nonnegative(variance, "metrics_raw.blur")
    return _clip(100.0 * (1.0 - math.log1p(min(v, 110.0)) / math.log1p(110.0)))


def noise_severity(linear_snr: float) -> float:
    """Draft linear-SNR formula; SNR is dimensionless, not dB."""
    if isinstance(linear_snr, bool) or not isinstance(linear_snr, (int, float)):
        raise ContractV2Error("finite SNR must be a non-negative number")
    r = float(linear_snr)
    if math.isinf(r) and r > 0:
        return 0.0
    if not math.isfinite(r) or r < 0:
        raise ContractV2Error("SNR must be finite/non-negative or verified positive infinity")
    return _clip(100.0 * (1.0 - r / 10.0))


def contrast_severity_parts(contrast: float) -> tuple[float, float, float]:
    """Return (combined, low-ramp, high-ramp); anchors 38/60 are provisional."""
    c = _finite_nonnegative(contrast, "metrics_raw.contrast")
    low = _clip(100.0 * (38.0 - c) / 18.0)
    high = _clip(100.0 * (c - 60.0) / 20.0)
    return max(low, high), low, high


def state_for_score(score: Any) -> str:
    if isinstance(score, bool) or not isinstance(score, int):
        raise ContractV2Error("fuzzy_score must be an integer in [0, 100]")
    if not 0 <= score <= 100:
        raise ContractV2Error("fuzzy_score must be an integer in [0, 100]")
    if score >= 75:
        return "ACCEPT"
    if score >= 45:
        return "BORDERLINE"
    return "REJECT"


def _source_snr_status(image: np.ndarray) -> tuple[str, float | None]:
    """Infer the SNR status from the image so 0/0 is not mislabeled infinity."""
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise ContractV2Error("source image must be a non-empty 2D grayscale array")
    pixels = image.astype(np.float64)
    mean = float(np.mean(pixels))
    std = float(np.std(pixels))
    if not math.isfinite(mean) or not math.isfinite(std) or mean < 0 or std < 0:
        raise ContractV2Error("source image produced invalid mean/std")
    if std == 0:
        if mean > 0:
            return "positive_infinity", None
        raise ContractV2Error("UNDEFINED_SNR: zero mean divided by zero standard deviation")
    snr = mean / std
    if not math.isfinite(snr) or snr < 0:
        raise ContractV2Error("source image produced invalid SNR")
    return "finite", snr


def build_v2_payload(m2_result: dict[str, Any], source_image: np.ndarray,
                     *, enable_high_contrast: bool = HIGH_CONTRAST_TRIGGER_ENABLED) -> dict[str, Any]:
    """Convert an actual M2 result to the draft v2 schema, or raise on inconsistency.

    `source_image` is required to distinguish true positive-infinite SNR from undefined
    0/0; a raw `inf` alone cannot establish that distinction.
    """
    if not isinstance(m2_result, dict):
        raise ContractV2Error("M2 result must be a dictionary")
    metrics = m2_result.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(RAW_KEYS):
        raise ContractV2Error("M2 metrics must contain exactly blur, contrast, noise")

    blur = _finite_nonnegative(metrics["blur"], "metrics_raw.blur")
    contrast = _finite_nonnegative(metrics["contrast"], "metrics_raw.contrast")
    status, image_snr = _source_snr_status(source_image)
    raw_noise = metrics["noise"]
    if status == "positive_infinity":
        if not (isinstance(raw_noise, (int, float)) and not isinstance(raw_noise, bool)
                and math.isinf(float(raw_noise)) and float(raw_noise) > 0):
            raise ContractV2Error("image has positive-infinite SNR but M2 noise value disagrees")
        raw_noise_json = None
        snr_for_severity = math.inf
    else:
        raw_noise_value = _finite_nonnegative(raw_noise, "metrics_raw.noise")
        assert image_snr is not None
        if not math.isclose(raw_noise_value, image_snr, rel_tol=1e-6, abs_tol=1e-8):
            raise ContractV2Error("M2 noise metric does not match source-image linear SNR")
        raw_noise_json = raw_noise_value
        snr_for_severity = raw_noise_value
        status = "finite"

    score = m2_result.get("fuzzy_score")
    state = state_for_score(score)
    source_state = m2_result.get("state")
    if source_state is not None and source_state != state:
        raise ContractV2Error("M2 state disagrees with fuzzy_score thresholds")

    blur_s = blur_severity(blur)
    contrast_s, low_contrast_s, high_contrast_s = contrast_severity_parts(contrast)
    noise_s = noise_severity(snr_for_severity)

    # Stable tie priority for enabled labels: motion_blur, low_contrast, sensor_noise.
    candidates = [
        (blur_s, "motion_blur"),
        (low_contrast_s, "low_contrast"),
        (noise_s, "sensor_noise"),
        (high_contrast_s, "high_contrast"),
    ]
    maximum = max(value for value, _ in candidates)
    if maximum == 0.0:
        if state != "ACCEPT":
            raise ContractV2Error("consistency error: non-ACCEPT state but all severities are zero")
        dominant = "none"
    else:
        if high_contrast_s > 0 and high_contrast_s >= max(blur_s, low_contrast_s, noise_s):
            if not enable_high_contrast:
                raise ContractV2Error(
                    "high_contrast would win/tie dominant selection but its production trigger is disabled"
                )
        # Exclude high_contrast until its trigger is explicitly enabled.
        enabled_candidates = candidates if enable_high_contrast else candidates[:3]
        dominant = next(label for value, label in enabled_candidates if value == maximum)

    payload = {
        "contract_version": CONTRACT_VERSION,
        "metrics_scale": METRICS_SCALE,
        "metrics_raw": {"blur": blur, "contrast": contrast, "noise": raw_noise_json},
        "metrics_raw_status": {
            "blur": "finite", "contrast": "finite", "noise": status
        },
        "metrics": {"blur": blur_s, "contrast": contrast_s, "noise": noise_s},
        "fuzzy_score": score,
        "state": state,
        "dominant_defect": dominant,
    }
    validate_v2_payload(payload, allow_high_contrast=enable_high_contrast)
    # Enforce standards-compliant JSON serialization before returning a success payload.
    json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return payload


def validate_v2_payload(payload: dict[str, Any], *,
                         allow_high_contrast: bool = HIGH_CONTRAST_TRIGGER_ENABLED) -> None:
    """Strictly validate the draft schema and its internal score/state/label rules."""
    required = {
        "contract_version", "metrics_scale", "metrics_raw", "metrics_raw_status",
        "metrics", "fuzzy_score", "state", "dominant_defect",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ContractV2Error("payload has missing or unexpected top-level fields")
    version = payload["contract_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ContractV2Error("contract_version must be integer 2")
    if payload["metrics_scale"] != METRICS_SCALE:
        raise ContractV2Error(f"metrics_scale must be {METRICS_SCALE!r}")
    raw = payload["metrics_raw"]
    statuses = payload["metrics_raw_status"]
    calibrated = payload["metrics"]
    if not isinstance(raw, dict) or set(raw) != set(RAW_KEYS):
        raise ContractV2Error("metrics_raw must contain exactly blur, contrast, noise")
    if not isinstance(statuses, dict) or set(statuses) != set(RAW_KEYS):
        raise ContractV2Error("metrics_raw_status must mirror metrics_raw keys")
    if not isinstance(calibrated, dict) or set(calibrated) != set(RAW_KEYS):
        raise ContractV2Error("metrics must contain exactly blur, contrast, noise")

    for key in ("blur", "contrast"):
        _finite_nonnegative(raw[key], f"metrics_raw.{key}")
        if statuses[key] != "finite":
            raise ContractV2Error(f"metrics_raw_status.{key} must be 'finite'")
    noise_status = statuses["noise"]
    if noise_status == "finite":
        _finite_nonnegative(raw["noise"], "metrics_raw.noise")
    elif noise_status == "positive_infinity":
        if raw["noise"] is not None:
            raise ContractV2Error("positive_infinity noise status requires JSON null raw value")
    else:
        raise ContractV2Error("noise status must be finite or positive_infinity")

    for key in RAW_KEYS:
        value = calibrated[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractV2Error(f"metrics.{key} must be numeric")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ContractV2Error(f"metrics.{key} must be finite and within [0, 100]")

    expected_state = state_for_score(payload["fuzzy_score"])
    if payload["state"] != expected_state:
        raise ContractV2Error("state disagrees with fuzzy_score thresholds")

    # Recompute calibrated metrics from the declared raw inputs so the serialized values
    # cannot drift from the published formula. Positive-infinite SNR is represented as null.
    expected_blur = blur_severity(raw["blur"])
    expected_contrast, low_contrast_severity, high_contrast_severity = contrast_severity_parts(raw["contrast"])
    expected_noise = 0.0 if noise_status == "positive_infinity" else noise_severity(raw["noise"])
    expected_metrics = {
        "blur": expected_blur,
        "contrast": expected_contrast,
        "noise": expected_noise,
    }
    for key, expected in expected_metrics.items():
        if not math.isclose(float(calibrated[key]), expected, rel_tol=0.0, abs_tol=1e-6):
            raise ContractV2Error(f"metrics.{key} disagrees with the documented provisional conversion")

    label = payload["dominant_defect"]
    if label not in LABELS:
        raise ContractV2Error("dominant_defect is not a canonical label")

    defect_candidates = [
        (expected_blur, "motion_blur"),
        (low_contrast_severity, "low_contrast"),
        (expected_noise, "sensor_noise"),
        (high_contrast_severity, "high_contrast"),
    ]
    maximum = max(score for score, _ in defect_candidates)
    if maximum == 0.0:
        if label != "none" or payload["state"] != "ACCEPT":
            raise ContractV2Error("all-zero severity requires dominant_defect none and ACCEPT state")
    else:
        if label == "none":
            raise ContractV2Error("none is invalid when any directional defect severity is positive")
        if high_contrast_severity > 0 and high_contrast_severity >= max(
            expected_blur, low_contrast_severity, expected_noise
        ) and not allow_high_contrast:
            raise ContractV2Error("high_contrast would win/tie dominant selection but its production trigger is disabled")
        enabled_candidates = defect_candidates if allow_high_contrast else defect_candidates[:3]
        expected_label = next(name for value, name in enabled_candidates if value == maximum)
        if label != expected_label:
            raise ContractV2Error("dominant_defect disagrees with highest directional severity/tie-break order")

    if all(float(calibrated[k]) == 0 for k in RAW_KEYS) and payload["state"] != "ACCEPT":
        raise ContractV2Error("non-ACCEPT state with all-zero severity is inconsistent")
