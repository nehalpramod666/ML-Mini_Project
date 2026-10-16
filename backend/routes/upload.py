import uuid
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from config import get_upload_dir

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only .dcm files are supported")

    upload_dir = get_upload_dir()
    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}.dcm"

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    return {"file_id": file_id, "filename": file.filename}