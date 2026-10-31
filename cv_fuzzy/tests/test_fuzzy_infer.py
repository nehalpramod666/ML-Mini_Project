from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pytest

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.fuzzy_engine import FuzzyInferenceError
from cv_fuzzy.fuzzy_infer import fuzzy_infer
from cv_fuzzy.v2_contract import ContractV2Error


def test_fuzzy_infer_returns_v2_payload_for_real_sample_dicom():
    dicom_path = Path(__file__).resolve().parent / "sample_data" / "CT_small.dcm"
    image = load_dicom(str(dicom_path))

    payload = fuzzy_infer(image)

    assert set(payload) == {
        "contract_version",
        "metrics_scale",
        "metrics_raw",
        "metrics_raw_status",
        "metrics",
        "fuzzy_score",
        "state",
        "dominant_defect",
    }

    assert payload["contract_version"] == 2
    assert payload["metrics_scale"] == "severity_0_100"

    assert set(payload["metrics_raw"]) == {"blur", "contrast", "noise"}
    assert set(payload["metrics_raw_status"]) == {"blur", "contrast", "noise"}
    assert set(payload["metrics"]) == {"blur", "contrast", "noise"}

    assert type(payload["fuzzy_score"]) is int
    assert 0 <= payload["fuzzy_score"] <= 100

    assert payload["state"] in {"ACCEPT", "BORDERLINE", "REJECT"}
    assert payload["dominant_defect"] in {
        "motion_blur",
        "low_contrast",
        "high_contrast",
        "low_snr",
        "none",
    }

    for severity in payload["metrics"].values():
        assert type(severity) is int
        assert 0 <= severity <= 100

    # The returned payload must be strictly JSON serializable.
    json.dumps(payload, allow_nan=False)


def test_fuzzy_infer_preserves_positive_infinite_snr_status():
    image = np.full((128, 128), 255, dtype=np.uint8)

    payload = fuzzy_infer(image)

    assert payload["metrics_raw"]["noise"] is None
    assert payload["metrics_raw_status"]["noise"] == "positive_infinity"
    assert payload["metrics"]["noise"] == 0

    json.dumps(payload, allow_nan=False)


def test_fuzzy_infer_preserves_undefined_snr_error():
    image = np.zeros((128, 128), dtype=np.uint8)

    with pytest.raises(ContractV2Error) as exc_info:
        fuzzy_infer(image)

    assert exc_info.value.code == "UNDEFINED_SNR"


def test_fuzzy_infer_preserves_disabled_high_contrast_error():
    image = np.array(
        [[0, 255], [255, 0]],
        dtype=np.uint8,
    )

    with pytest.raises(ContractV2Error) as exc_info:
        fuzzy_infer(image)

    assert exc_info.value.code == "DISABLED_HIGH_CONTRAST_TRIGGER"


def test_fuzzy_infer_rejects_non_numpy_input():
    with pytest.raises(TypeError):
        fuzzy_infer("not an image")


def test_fuzzy_infer_rejects_color_image():
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        fuzzy_infer(image)


def test_fuzzy_infer_rejects_empty_image():
    image = np.empty((0, 0), dtype=np.uint8)

    with pytest.raises(ValueError):
        fuzzy_infer(image)


def test_fuzzy_infer_preserves_fuzzy_inference_error(monkeypatch):
    module = import_module("cv_fuzzy.fuzzy_infer")
    expected_error = FuzzyInferenceError("simulated fuzzy inference failure")

    def raise_expected_error(_image):
        raise expected_error

    monkeypatch.setattr(
        module,
        "analyze_image_fuzzy",
        raise_expected_error,
    )

    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(FuzzyInferenceError) as exc_info:
        module.fuzzy_infer(image)

    assert exc_info.value is expected_error
