from fastapi import APIRouter, HTTPException
import os

router = APIRouter()

UPLOAD_DIR = "temp_uploads"

# Hardcoded mock — matches the contract in docs/contract.md exactly.
# This will be replaced with a real call to M2's fuzzy engine
# and M1's LLM engine once those are ready (Week 6-8).
MOCK_RESPONSE = {
    "metrics": {
        "blur": 85.2,
        "contrast": 40.1,
        "noise": 12.4
    },
    "fuzzy_score": 32,
    "state": "REJECT",
    "dominant_defect": "motion_blur"
}

@router.post("/analyze/{file_id}")
async def analyze_file(file_id: str):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.dcm")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # TODO (Week 6-8): replace MOCK_RESPONSE with real cv_fuzzy.analyze_image(file_path)
    return MOCK_RESPONSE