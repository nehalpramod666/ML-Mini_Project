import asyncio
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import get_max_upload_bytes, get_upload_dir
from dicom_validation import UploadValidationError, validate_dicom_file

router = APIRouter()

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".dcm"):
        raise _error(400, "INVALID_FILE_TYPE", "Only .dcm files are supported")

    upload_dir = get_upload_dir()
    max_bytes = get_max_upload_bytes()
    file_id = str(uuid.uuid4())
    file_path = upload_dir / f"{file_id}.dcm"

    size = 0
    try:
        with open(file_path, "wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise _error(
                        413,
                        "FILE_TOO_LARGE",
                        f"File exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    )
                out.write(chunk)

        if size == 0:
            raise _error(400, "EMPTY_FILE", "The uploaded file is empty")

        await asyncio.to_thread(validate_dicom_file, file_path)
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except UploadValidationError as exc:
        file_path.unlink(missing_ok=True)
        raise _error(exc.status_code, exc.code, exc.message) from exc
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise _error(500, "UPLOAD_FAILED", "Failed to save uploaded file") from exc

    return {"file_id": file_id, "filename": file.filename}