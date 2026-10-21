"""Integration tests for the real analysis pipeline in backend/routes/analyze.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import app
from cv_fuzzy.fuzzy_engine import FuzzyInferenceError
from cv_fuzzy.v2_contract import ContractV2Error, validate_v2_payload
from ml.llm_engine import ContractError, ModelUnavailableError

client = TestClient(app)

SAMPLE_DICOM = Path(__file__).resolve().parents[2] / "cv_fuzzy" / "sample_data" / "CT_small.dcm"


@pytest.fixture
def test_upload_dir(tmp_path, monkeypatch):
    """Ensure upload tests run in an isolated temporary directory."""
    monkeypatch.setenv("GATEKEEPER_UPLOAD_DIR", str(tmp_path))
    return tmp_path


def test_successful_dicom_analysis_real_pipeline(test_upload_dir, monkeypatch):
    """Exercise load_dicom -> analyze_image_fuzzy -> build_v2_payload -> mocked generate_report."""
    assert SAMPLE_DICOM.is_file(), f"Sample DICOM not found at {SAMPLE_DICOM}"

    # 1. Upload the real DICOM
    dicom_bytes = SAMPLE_DICOM.read_bytes()
    upload_res = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", dicom_bytes, "application/octet-stream")},
    )
    assert upload_res.status_code == 200
    file_id = upload_res.json()["file_id"]

    # 2. Mock LLM inference boundary
    mock_report_text = "Clinical QA: Retake required due to low signal-to-noise ratio."
    captured_payloads = []

    def mock_generate_report(payload: dict) -> str:
        captured_payloads.append(payload)
        return mock_report_text

    monkeypatch.setattr("routes.analyze.generate_report", mock_generate_report)

    # 3. Analyze
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"assessment", "report"}
    assert data["report"] == mock_report_text

    # 4. Validate exact 8-field v2 assessment schema
    assessment = data["assessment"]
    expected_fields = {
        "contract_version",
        "metrics_scale",
        "metrics_raw",
        "metrics_raw_status",
        "metrics",
        "fuzzy_score",
        "state",
        "dominant_defect",
    }
    assert set(assessment.keys()) == expected_fields
    assert assessment["contract_version"] == 2
    assert assessment["metrics_scale"] == "severity_0_100"

    # 5. Metrics must be integers in [0, 100]
    metrics = assessment["metrics"]
    for key in ("blur", "contrast", "noise"):
        assert isinstance(metrics[key], int) and not isinstance(metrics[key], bool)
        assert 0 <= metrics[key] <= 100

    # 6. Must strictly pass v2 validator
    validate_v2_payload(assessment)

    # 7. Verify LLM received exact assessment payload
    assert len(captured_payloads) == 1
    assert captured_payloads[0] == assessment


def test_analyze_missing_file_returns_404(test_upload_dir):
    response = client.post("/analyze/non-existent-uuid-1234")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["code"] == "FILE_NOT_FOUND"
    assert data["detail"]["message"] == "File not found"


def test_analyze_path_traversal_returns_404(test_upload_dir):
    response = client.post("/analyze/test..file")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "FILE_NOT_FOUND"
    assert response.json()["detail"]["message"] == "File not found"


def test_analyze_malformed_dicom_returns_422_without_calling_llm(test_upload_dir, monkeypatch):
    # /upload now rejects non-DICOM bytes, so place the corrupt file directly
    # in the upload folder to test how /analyze handles it.
    file_id = "corrupt-file"
    (test_upload_dir / f"{file_id}.dcm").write_bytes(b"NOT_A_DICOM_FILE_HEADER")

    mock_llm = MagicMock()
    monkeypatch.setattr("routes.analyze.generate_report", mock_llm)

    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["code"] == "INVALID_DICOM"
    assert data["detail"]["message"] == "The uploaded file could not be read as a valid DICOM image."
    # Ensure sensitive internal paths or raw file header details are not leaked
    assert "corrupt-file" not in response.text
    assert "NOT_A_DICOM" not in response.text
    mock_llm.assert_not_called()

def test_analyze_fuzzy_inference_failure_returns_422(test_upload_dir, monkeypatch):
    file_id = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", SAMPLE_DICOM.read_bytes(), "application/octet-stream")},
    )
    file_id = file_id.json()["file_id"]

    sensitive_detail = "Sensitive rule gap at C:\\hospital\\internal_config.yaml"

    def mock_fuzzy_fail(image):
        raise FuzzyInferenceError(sensitive_detail)

    monkeypatch.setattr("routes.analyze.analyze_image_fuzzy", mock_fuzzy_fail)

    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["code"] == "FUZZY_INFERENCE_ERROR"
    assert data["detail"]["message"] == "Image quality assessment could not be completed."
    assert "internal_config" not in response.text
    assert "Sensitive rule gap" not in response.text


def test_analyze_unexpected_fuzzy_failure_returns_422_assessment_failed(test_upload_dir, monkeypatch):
    upload_res = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", SAMPLE_DICOM.read_bytes(), "application/octet-stream")},
    )
    file_id = upload_res.json()["file_id"]

    sensitive_detail = "Core dump at C:\\system\\debug\\fuzzy_matrix_dump.bin"

    def mock_fuzzy_crash(image):
        raise ZeroDivisionError(sensitive_detail)

    monkeypatch.setattr("routes.analyze.analyze_image_fuzzy", mock_fuzzy_crash)

    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["code"] == "ASSESSMENT_FAILED"
    assert data["detail"]["message"] == "Image quality assessment failed."
    assert "fuzzy_matrix_dump" not in response.text
    assert "ZeroDivisionError" not in response.text


def test_analyze_contract_v2_error_preserves_stable_code(test_upload_dir, monkeypatch):
    upload_res = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", SAMPLE_DICOM.read_bytes(), "application/octet-stream")},
    )
    file_id = upload_res.json()["file_id"]

    # Test DISABLED_HIGH_CONTRAST_TRIGGER
    def mock_high_contrast_fail(m2_res, img):
        raise ContractV2Error(
            "High contrast triggered at internal memory 0xdeadbeef",
            code="DISABLED_HIGH_CONTRAST_TRIGGER",
        )

    monkeypatch.setattr("routes.analyze.build_v2_payload", mock_high_contrast_fail)
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["code"] == "DISABLED_HIGH_CONTRAST_TRIGGER"
    assert data["detail"]["message"] == "High-contrast defect trigger is currently disabled."
    assert "0xdeadbeef" not in response.text

    # Test CONSISTENCY_ERROR
    def mock_consistency_fail(m2_res, img):
        raise ContractV2Error(
            "Detailed internal mismatch in C:\\secret\\rules.py: score=20 vs 75",
            code="CONSISTENCY_ERROR",
        )

    monkeypatch.setattr("routes.analyze.build_v2_payload", mock_consistency_fail)
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["code"] == "CONSISTENCY_ERROR"
    assert data["detail"]["message"] == "Image quality assessment is internally inconsistent."
    assert "secret" not in response.text
    assert "Detailed internal mismatch" not in response.text


def test_analyze_contract_adapter_error_returns_500(test_upload_dir, monkeypatch):
    upload_res = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", SAMPLE_DICOM.read_bytes(), "application/octet-stream")},
    )
    file_id = upload_res.json()["file_id"]

    def mock_adapter_fail(payload):
        raise ContractError("Internal prompt formatting error on line 42 of C:\\vault\\engine.py")

    monkeypatch.setattr("routes.analyze.generate_report", mock_adapter_fail)
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 500
    data = response.json()
    assert data["detail"]["code"] == "CONTRACT_ADAPTER_ERROR"
    assert data["detail"]["message"] == "The assessment could not be processed for report generation."
    assert "vault" not in response.text
    assert "formatting error" not in response.text


def test_analyze_model_unavailable_returns_503(test_upload_dir, monkeypatch):
    upload_res = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", SAMPLE_DICOM.read_bytes(), "application/octet-stream")},
    )
    file_id = upload_res.json()["file_id"]

    def mock_model_unavailable(payload):
        raise ModelUnavailableError("GGUF model file not found at C:\\models\\internal.gguf")

    monkeypatch.setattr("routes.analyze.generate_report", mock_model_unavailable)
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 503
    data = response.json()
    assert data["detail"]["code"] == "MODEL_UNAVAILABLE"
    assert data["detail"]["message"] == "Clinical QA model is currently unavailable"
    assert "internal.gguf" not in response.text


def test_analyze_unexpected_report_failure_returns_500(test_upload_dir, monkeypatch):
    upload_res = client.post(
        "/upload",
        files={"file": ("CT_small.dcm", SAMPLE_DICOM.read_bytes(), "application/octet-stream")},
    )
    file_id = upload_res.json()["file_id"]

    def mock_unexpected_fail(payload):
        raise RuntimeError("Critical crash in /opt/models/weights/clinical.gguf with traceback 0x01")

    monkeypatch.setattr("routes.analyze.generate_report", mock_unexpected_fail)
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 500
    data = response.json()
    assert data["detail"]["code"] == "REPORT_GENERATION_FAILED"
    assert data["detail"]["message"] == "Failed to generate clinical report"
    assert "/opt/models" not in response.text
    assert "traceback" not in response.text
