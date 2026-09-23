from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)

UPLOAD_DIR = "temp_uploads"


def test_analyze_returns_mock_contract_for_valid_file():
    # First upload a fake .dcm file to get a real file_id
    upload_response = client.post(
        "/upload",
        files={"file": ("sample.dcm", b"fake dicom bytes", "application/octet-stream")}
    )
    file_id = upload_response.json()["file_id"]

    # Now analyze it
    response = client.post(f"/analyze/{file_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["metrics"]["blur"] == 85.2
    assert data["metrics"]["contrast"] == 40.1
    assert data["metrics"]["noise"] == 12.4
    assert data["fuzzy_score"] == 32
    assert data["state"] == "REJECT"
    assert data["dominant_defect"] == "motion_blur"


def test_analyze_returns_404_for_missing_file():
    response = client.post("/analyze/does-not-exist-123")
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"