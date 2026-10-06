import json
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NVIDIA DLL directories installed inside the virtual environment
VENV = Path(os.environ["VIRTUAL_ENV"])

NVIDIA_DLL_DIRS = [
    VENV / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
    VENV / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
]

# Add NVIDIA DLL directories to PATH before importing llama_cpp.
for dll_dir in NVIDIA_DLL_DIRS:
    if dll_dir.exists():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]

# Keep the DLL directory handles alive for the lifetime of the process.
_dll_handles = []
for dll_dir in NVIDIA_DLL_DIRS:
    if dll_dir.exists():
        _dll_handles.append(os.add_dll_directory(str(dll_dir)))

from llama_cpp import Llama


# Model is currently stored in the repository's root models/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "clinical-gatekeeper-Q4_K_M.gguf"


llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False,
)


def generate_report(metrics_json: dict) -> str:
    """
    Generate a concise image-quality report from deterministic metrics.

    Args:
        metrics_json: Dictionary containing:
            blur, contrast, noise, state, dominant_defect

    Returns:
        A concise quality report as a string.
    """

    system_prompt = (
        "You are a clinical imaging quality assistant. "
        "You receive deterministic DICOM image-quality metrics. "
        "Explain the dominant image-quality defect and give a concise, "
        "actionable retake instruction. "
        "Do not invent patient details, image numbers, body parts, "
        "measurements, or clinical findings. "
        "Use only the information provided. "
        "Do not describe the image as if you can see it."
    )

    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{json.dumps(metrics_json)}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = llm(
        prompt,
        max_tokens=100,
        temperature=0.0,
        stop=["<|eot_id|>"],
    )

    return output["choices"][0]["text"].strip()


if __name__ == "__main__":
    test_metrics = {
        "blur": 85.2,
        "contrast": 40.1,
        "noise": 12.4,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }

    print("=== MODEL OUTPUT ===")
    print(generate_report(test_metrics))