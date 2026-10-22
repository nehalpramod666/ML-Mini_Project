"""Serve an uploaded DICOM back to the browser viewer."""
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import get_upload_dir

router = APIRouter()

SAFE_FILE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "FILE_NOT_FOUND", "message": "File not found"},
    )


@router.get("/files/{file_id}")
async def get_file(file_id: str):
    if not SAFE_FILE_ID_REGEX.fullmatch(file_id):
        raise _not_found()

    upload_dir = get_upload_dir()
    file_path = (upload_dir / f"{file_id}.dcm").resolve()

    try:
        file_path.relative_to(upload_dir)
    except ValueError:
        raise _not_found()

    if not file_path.is_file():
        raise _not_found()

    return FileResponse(
        file_path,
        media_type="application/dicom",
        headers={"Cache-Control": "no-store"},
    )