from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from main import app

client = TestClient(app)

def test_upload_rejects_non_dcm_file():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only .dcm files are supported"

def test_upload_accepts_dcm_file():
    response = client.post(
        "/upload",
        files={"file": ("sample.dcm", b"fake dicom bytes", "application/octet-stream")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "sample.dcm"