"""Analysis route integrating DICOM loading, fuzzy QA assessment, and LLM report generation."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config import get_upload_dir
from cv_fuzzy.dicom_loader import DicomLoadError, load_dicom
from cv_fuzzy.fuzzy_engine import FuzzyInferenceError, analyze_image_fuzzy
from cv_fuzzy.v2_contract import ContractV2Error, build_v2_payload
from ml.llm_engine import ContractError, ModelUnavailableError, generate_report

logger = logging.getLogger(__name__)

router = APIRouter()

SAFE_FILE_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")

CONTRACT_ERROR_MESSAGES = {
    "DISABLED_HIGH_CONTRAST_TRIGGER": "High-contrast defect trigger is currently disabled.",
    "CONSISTENCY_ERROR": "Image quality assessment is internally inconsistent.",
    "UNDEFINED_SNR": "Signal-to-noise ratio could not be determined.",
    "INVALID_SCHEMA": "Quality assessment schema is invalid.",
}


def _execute_pipeline(file_path: Path) -> dict:
    """Synchronous pipeline executed in a worker thread."""
    # 1. Load DICOM
    try:
        image = load_dicom(str(file_path))
    except DicomLoadError as exc:
        logger.warning("DICOM loading failed for %s: %s", file_path.name, exc)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_DICOM",
                "message": "The uploaded file could not be read as a valid DICOM image.",
            },
        ) from exc
    except Exception as exc:
        logger.warning("Unexpected error loading DICOM %s: %s", file_path.name, exc, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_DICOM",
                "message": "The uploaded file could not be read as a valid DICOM image.",
            },
        ) from exc

    # 2. Fuzzy assessment
    try:
        m2_result = analyze_image_fuzzy(image)
    except FuzzyInferenceError as exc:
        logger.error("Fuzzy inference error for %s: %s", file_path.name, exc)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "FUZZY_INFERENCE_ERROR",
                "message": "Image quality assessment could not be completed.",
            },
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error in fuzzy analysis for %s: %s", file_path.name, exc, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "ASSESSMENT_FAILED",
                "message": "Image quality assessment failed.",
            },
        ) from exc

    # 3. Build validated v2 payload
    try:
        assessment_payload = build_v2_payload(m2_result, image)
    except ContractV2Error as exc:
        logger.error("Contract v2 error for %s: %s", file_path.name, exc)
        error_code = exc.code or "CONTRACT_V2_ERROR"
        safe_message = CONTRACT_ERROR_MESSAGES.get(
            error_code,
            "Quality assessment contract validation failed.",
        )
        raise HTTPException(
            status_code=422,
            detail={"code": error_code, "message": safe_message},
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error constructing contract v2 payload for %s: %s", file_path.name, exc, exc_info=True)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "CONSISTENCY_ERROR",
                "message": "Image quality assessment is internally inconsistent.",
            },
        ) from exc

    # 4. LLM report generation
    try:
        report_text = generate_report(assessment_payload)
    except ContractError as exc:
        logger.error("Contract adapter error generating report for %s: %s", file_path.name, exc)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CONTRACT_ADAPTER_ERROR",
                "message": "The assessment could not be processed for report generation.",
            },
        ) from exc
    except (ModelUnavailableError, FileNotFoundError) as exc:
        logger.error("Model unavailable during report generation: %s", exc)
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": "Clinical QA model is currently unavailable",
            },
        ) from exc
    except Exception as exc:
        logger.error("Report generation failed for %s: %s", file_path.name, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "REPORT_GENERATION_FAILED",
                "message": "Failed to generate clinical report",
            },
        ) from exc

    return {
        "assessment": assessment_payload,
        "report": report_text,
    }


@router.post("/analyze/{file_id}")
async def analyze_file(file_id: str):
    if not SAFE_FILE_ID_REGEX.fullmatch(file_id):
        raise HTTPException(
            status_code=404,
            detail={"code": "FILE_NOT_FOUND", "message": "File not found"},
        )

    upload_dir = get_upload_dir()
    file_path = (upload_dir / f"{file_id}.dcm").resolve()

    try:
        file_path.relative_to(upload_dir)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail={"code": "FILE_NOT_FOUND", "message": "File not found"},
        )

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail={"code": "FILE_NOT_FOUND", "message": "File not found"},
        )

    return await asyncio.to_thread(_execute_pipeline, file_path)