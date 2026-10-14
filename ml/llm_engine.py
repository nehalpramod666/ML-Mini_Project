
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

# Labels currently emitted by M2's fuzzy engine.
M2_DEFECTS = {
    "motion_blur",
    "low_contrast",
    "high_noise",
    "none",
}

_llm = None
_load_lock = threading.Lock()
_gen_lock = threading.Lock()

# Keep Windows DLL-directory handles alive while inference is running.
_dll_handles = []


class ContractError(ValueError):
    """Input is invalid or incompatible with the model's input schema."""


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
                    raise FileNotFoundError(
                        f"GGUF model not found: {MODEL_PATH}. "
                        "Set GATEKEEPER_GGUF to the model's actual path."
                    )

                _prepare_runtime_dlls()

                try:
                    from llama_cpp import Llama
                except ImportError as exc:
                    raise RuntimeError(
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


def to_llm_input(contract: dict) -> dict:
    """
    Validate an input and return the exact flat shape used during training.

    Nested version-2 contracts must explicitly declare calibrated
    severity_0_100 scores. Raw M2 measurements are never passed directly
    to the LLM as if they were calibrated scores.

    The high_noise -> sensor_noise mapping remains blocked pending team
    agreement on canonical labels.
    """
    if not isinstance(contract, dict):
        raise ContractError("Input must be a JSON object")

    if "metrics" in contract:
        # Reject legacy/raw nested contracts before interpreting metric values.
        if contract.get("metrics_scale") != SUPPORTED_SCALE:
            raise ContractError(
                f"metrics_scale={contract.get('metrics_scale')!r}; "
                f"expected {SUPPORTED_SCALE!r}. "
                "Raw metrics must be calibrated by the CV/fuzzy engine, not here."
            )

        if contract.get("contract_version") != 2:
            raise ContractError("Calibrated nested input requires contract_version=2")

        raw_metrics = contract.get("metrics_raw")
        if not isinstance(raw_metrics, dict):
            raise ContractError("Version-2 contract requires a metrics_raw object")

        # Raw metrics are retained for traceability, never sent to the model.
        # Use JSON null for a non-finite raw value before serialization.
        for field in ("blur", "contrast", "noise"):
            if field not in raw_metrics:
                raise ContractError(f"Missing raw metric: metrics_raw.{field}")

            raw_value = raw_metrics[field]
            if raw_value is not None:
                finite_raw = _finite_number(
                    raw_value, f"metrics_raw.{field}"
                )
                if finite_raw < 0:
                    raise ContractError(
                        f"metrics_raw.{field} cannot be negative"
                    )

        metrics = contract["metrics"]
        if not isinstance(metrics, dict):
            raise ContractError("'metrics' must contain calibrated scores")

        try:
            flat = {
                "blur": round(_finite_number(metrics["blur"], "metrics.blur"), 1),
                "contrast": round(
                    _finite_number(metrics["contrast"], "metrics.contrast"), 1
                ),
                "noise": round(_finite_number(metrics["noise"], "metrics.noise"), 1),
                "state": contract["state"],
                "dominant_defect": contract["dominant_defect"],
            }
        except (KeyError, TypeError, AttributeError) as exc:
            raise ContractError(
                f"Malformed calibrated contract: {exc!r}"
            ) from exc

        fuzzy_score = contract.get("fuzzy_score")
        if (
            isinstance(fuzzy_score, bool)
            or not isinstance(fuzzy_score, int)
            or not 0 <= fuzzy_score <= 100
        ):
            raise ContractError("fuzzy_score must be an integer from 0 to 100")

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
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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
