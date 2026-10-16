
"""
Local GGUF inference for clinical retake instructions.

Validated hardware configuration:
    Model: clinical-gatekeeper-Q4_K_M.gguf
    Context: 1024 tokens
    GPU offload: all layers
    Target: RTX 3050 Laptop GPU, 4 GB VRAM

Deployment requirements:
    - Run one application process/worker only on the 4 GB GPU.
    - Call warmup() once during FastAPI startup/lifespan.
    - Run blocking inference in a worker thread, not on the async event loop.

IMPORTANT:
    The LLM was trained on synthetic 0-100 input scores. M2 currently supplies
    raw CV measurements. Nested M2 contracts are validated but rejected for
    inference until an agreed, calibrated raw-to-score adapter exists.
"""

import json
import logging
import math
import os
import sys
import threading
from pathlib import Path
from typing import Iterator

from cv_fuzzy.v2_contract import ContractV2Error, validate_v2_payload

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.environ.get(
        "GATEKEEPER_GGUF",
        str(PROJECT_ROOT / "models" / "clinical-gatekeeper-Q4_K_M.gguf"),
    )
).expanduser()

N_CTX = 1024
MAX_TOKENS = 60
TEMPERATURE = 0.0
SUPPORTED_SCALE = "severity_0_100"

SYSTEM_PROMPT = (
    "You are a clinical imaging quality assistant. Given DICOM quality metrics "
    "(blur, contrast, noise scores, overall state, and dominant defect), explain "
    "the defect and give a concise, actionable retake instruction."
)

VALID_STATES = {"ACCEPT", "BORDERLINE", "REJECT"}

# Labels present in the model's synthetic training schema.
TRAINING_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_contrast",
    "sensor_noise",
    "none",
}

# Canonical labels defined for the v2 contract.
CANONICAL_V2_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_contrast",
    "low_snr",
    "none",
}

# Labels currently emitted by legacy M2 fuzzy engine.
M2_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_noise",
    "none",
}

SNR_CONTEXTUAL_INSTRUCTION = (
    "Note: The defect category 'sensor_noise' reflects a low signal-to-noise ratio "
    "(SNR) measurement or proxy, not confirmed physical sensor hardware malfunction. "
    "Describe the issue neutrally as an SNR or noise-related quality observation."
)

_llm = None
_load_lock = threading.Lock()
_gen_lock = threading.Lock()

# Keep Windows DLL-directory handles alive while inference is running.
_dll_handles = []


class ContractError(ValueError):
    """Input is invalid or incompatible with the model's input schema."""


class ModelUnavailableError(RuntimeError):
    """Raised when the GGUF model file or inference dependency is unavailable."""


def _prepare_runtime_dlls() -> None:
    """Make CUDA DLL directories discoverable on Windows when available."""
    if os.name != "nt":
        return

    venv = Path(sys.prefix)
    dll_dirs = [
        venv / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        venv / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
    ]

    for dll_dir in dll_dirs:
        if not dll_dir.is_dir():
            continue

        os.environ["PATH"] = (
            str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        )

        if hasattr(os, "add_dll_directory"):
            _dll_handles.append(os.add_dll_directory(str(dll_dir)))


def get_llm():
    """
    Load the model on first use.

    Importing this module does not import llama_cpp or load the GGUF.
    The lock prevents concurrent requests from initializing multiple copies.
    """
    global _llm

    if _llm is None:
        with _load_lock:
            if _llm is None:
                if not MODEL_PATH.is_file():
                    raise ModelUnavailableError(
                        f"GGUF model not found: {MODEL_PATH}. "
                        "Set GATEKEEPER_GGUF to the model's actual path."
                    )

                _prepare_runtime_dlls()

                try:
                    from llama_cpp import Llama
                except ImportError as exc:
                    raise ModelUnavailableError(
                        "llama-cpp-python is required for local inference."
                    ) from exc

                logger.info("Loading GGUF model from %s", MODEL_PATH)

                _llm = Llama(
                    model_path=str(MODEL_PATH),
                    n_ctx=N_CTX,
                    n_gpu_layers=-1,
                    verbose=False,
                )

                logger.info("GGUF model loaded successfully")

    return _llm


def _finite_number(value, field: str) -> float:
    """Accept only finite JSON-style numbers; reject bool, NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field} must be a finite number")

    number = float(value)

    if not math.isfinite(number):
        raise ContractError(f"{field} must be finite; got {value!r}")

    return number


def _validate_severity_int(value, field: str) -> int:
    """Accept only integer severities in [0, 100]; reject booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{field} must be an integer; got {value!r}")

    if not 0 <= value <= 100:
        raise ContractError(f"{field}={value} outside calibrated [0, 100] range")

    return value


def to_llm_input(contract: dict) -> dict:
    """
    Validate an input and return the exact flat shape used during training.

    Nested version-2 contracts must explicitly declare calibrated
    severity_0_100 scores as integers. Raw M2 measurements are never passed
    directly to the LLM as if they were calibrated scores.

    Phase 1 Adapter:
    - Canonical v2 label 'low_snr' is explicitly mapped to the model's training
      vocabulary token 'sensor_noise'.
    - 'sensor_noise' is rejected as a canonical label in nested v2 payloads.
    - Legacy 'high_noise' is strictly rejected.
    """
    if not isinstance(contract, dict):
        raise ContractError("Input must be a JSON object")

    if "metrics" in contract or "metrics_raw" in contract or "contract_version" in contract or "metrics_scale" in contract:
        # Reject legacy/raw nested contracts before interpreting metric values.
        if contract.get("metrics_scale") != SUPPORTED_SCALE:
            raise ContractError(
                f"metrics_scale={contract.get('metrics_scale')!r}; "
                f"expected {SUPPORTED_SCALE!r}. "
                "Raw metrics must be calibrated by the CV/fuzzy engine, not here."
            )

        raw_defect = contract.get("dominant_defect")
        if raw_defect == "sensor_noise":
            raise ContractError(
                "Nested v2 contract must use canonical label 'low_snr', not internal training label 'sensor_noise'"
            )
        if raw_defect == "high_noise":
            raise ContractError(
                "Legacy label 'high_noise' has been rejected and cannot be mapped"
            )

        try:
            validate_v2_payload(contract)
        except ContractV2Error as exc:
            raise ContractError(f"Invalid v2 contract payload: {exc}") from exc

        adapted_defect = "sensor_noise" if raw_defect == "low_snr" else raw_defect
        metrics = contract["metrics"]
        flat = {
            "blur": metrics["blur"],
            "contrast": metrics["contrast"],
            "noise": metrics["noise"],
            "state": contract["state"],
            "dominant_defect": adapted_defect,
        }

    else:
        # Existing flat input used by training/evaluation scripts.
        expected = {
            "blur", "contrast", "noise", "state", "dominant_defect"
        }
        if set(contract) != expected:
            missing = expected - contract.keys()
            extra = contract.keys() - expected
            raise ContractError(
                f"Invalid flat model input; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        flat = {
            "blur": round(_finite_number(contract["blur"], "blur"), 1),
            "contrast": round(_finite_number(contract["contrast"], "contrast"), 1),
            "noise": round(_finite_number(contract["noise"], "noise"), 1),
            "state": contract["state"],
            "dominant_defect": contract["dominant_defect"],
        }

    for field in ("blur", "contrast", "noise"):
        if not 0.0 <= flat[field] <= 100.0:
            raise ContractError(
                f"{field}={flat[field]} outside the model's trained "
                "0-100 input range"
            )

    if flat["state"] not in VALID_STATES:
        raise ContractError(f"Invalid state: {flat['state']!r}")

    if flat["dominant_defect"] not in TRAINING_DEFECTS:
        if flat["dominant_defect"] in M2_DEFECTS:
            raise ContractError(
                f"M2 defect label {flat['dominant_defect']!r} has not been "
                "approved for mapping to the model's training labels"
            )
        raise ContractError(
            f"Invalid training dominant_defect: "
            f"{flat['dominant_defect']!r}"
        )

    return flat


def _messages(flat: dict) -> list:
    """Build chat messages; the GGUF's stored chat template formats them."""
    system_content = SYSTEM_PROMPT
    if flat.get("dominant_defect") == "sensor_noise":
        system_content = f"{SYSTEM_PROMPT}\n\n{SNR_CONTEXTUAL_INSTRUCTION}"
    return [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": json.dumps(flat, allow_nan=False),
        },
    ]



def generate_report(contract: dict) -> str:
    """
    Generate one report using the trained flat input schema.

    Raises ContractError for invalid input or uncalibrated raw M2 contracts.
    Model loading and inference are blocking operations.
    """
    flat = to_llm_input(contract)
    model = get_llm()

    # Serialize access: a Llama instance must not serve overlapping calls.
    with _gen_lock:
        response = model.create_chat_completion(
            messages=_messages(flat),
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

    choice = response["choices"][0]

    if choice.get("finish_reason") == "length":
        logger.warning(
            "Report truncated at max_tokens=%d for input %s",
            MAX_TOKENS,
            flat,
        )

    return (choice.get("message", {}).get("content") or "").strip()


def generate_report_stream(contract: dict) -> Iterator[str]:
    """
    Return an iterator of generated text chunks.

    Validation happens when this function is called. The generation lock
    remains held for the stream's lifetime to avoid concurrent model use.
    """
    flat = to_llm_input(contract)
    model = get_llm()

    def iterate() -> Iterator[str]:
        with _gen_lock:
            stream = model.create_chat_completion(
                messages=_messages(flat),
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )

            for chunk in stream:
                choice = chunk["choices"][0]

                if choice.get("finish_reason") == "length":
                    logger.warning(
                        "Streamed report truncated at max_tokens=%d for input %s",
                        MAX_TOKENS,
                        flat,
                    )

                text = choice.get("delta", {}).get("content")

                if text:
                    yield text

    return iterate()


def warmup() -> None:
    """
    Load the model and run one short inference during backend startup.

    Call from a FastAPI lifespan/startup hook executed in a worker thread.
    """
    generate_report(
        {
            "blur": 0.0,
            "contrast": 50.0,
            "noise": 0.0,
            "state": "ACCEPT",
            "dominant_defect": "none",
        }
    )


if __name__ == "__main__":
    test_metrics = {
        "blur": 85.2,
        "contrast": 40.1,
        "noise": 12.4,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }

    print("--- Blocking ---")
    print(generate_report(test_metrics))

    print("\n--- Streaming ---")
    print("".join(generate_report_stream(test_metrics)))
