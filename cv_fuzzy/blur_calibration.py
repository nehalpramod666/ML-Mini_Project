"""Run a separate Gaussian-blur calibration experiment."""

import json
from pathlib import Path

import cv2

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.metrics import compute_blur


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cv_fuzzy" / "sample_data" / "CT_small.dcm"
OUTPUT_DIR = ROOT / "cv_fuzzy" / "blur_calibration_results"

CANDIDATE_SETTINGS = [
    (3, 0.5),
    (5, 0.8),
    (5, 1.0),
    (9, 2.0),
    (15, 5.0),
]


def run_calibration():
    """Save candidate blur images and their measured scores."""
    original = load_dicom(str(SOURCE))

    if original.ndim != 2:
        raise ValueError("Expected a 2D grayscale source image")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_score = compute_blur(original)
    records = []

    for kernel_size, sigma in CANDIDATE_SETTINGS:
        degraded = cv2.GaussianBlur(
            original,
            (kernel_size, kernel_size),
            sigmaX=sigma,
        )

        score = compute_blur(degraded)
        filename = f"blur_k{kernel_size:02d}_s{str(sigma).replace('.', 'p')}.png"
        output_path = OUTPUT_DIR / filename

        if not cv2.imwrite(str(output_path), degraded):
            raise RuntimeError(f"Could not save {output_path}")

        records.append({
            "filename": filename,
            "source": str(SOURCE.relative_to(ROOT)),
            "kernel_size": kernel_size,
            "sigma": sigma,
            "original_blur_score": original_score,
            "blur_score": score,
        })

        print(
            f"{filename}: score={score:.4f}, "
            f"change={original_score - score:.4f}"
        )

    manifest_path = OUTPUT_DIR / "calibration_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nOriginal score: {original_score:.4f}")
    print(f"Calibration images: {len(records)}")
    print(f"Calibration manifest: {manifest_path}")


if __name__ == "__main__":
    run_calibration()
