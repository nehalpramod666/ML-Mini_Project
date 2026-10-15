import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.fuzzy_engine import analyze_image_fuzzy
from cv_fuzzy.impulse_noise import estimate_impulse_noise
from cv_fuzzy.v2_contract import (
    build_v2_payload,
    contrast_severity_parts,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gold_standard"
SOURCE = ROOT / "cv_fuzzy" / "sample_data" / "CT_small.dcm"
MANIFEST = DATA / "manifest.json"
OUT_JSON = ROOT / "reports" / "m2_day14_candidate_evaluation.json"
OUT_CSV = ROOT / "reports" / "m2_day14_candidate_per_image.csv"

# Exploratory only; do not use this as production calibration.
IMPULSE_THRESHOLD = 0.001  # 0.1% of pixels
SEVERITY_FLOORS = (0.0, 1.0, 5.0, 10.0)
EXPECTED = {
    "blur": "motion_blur",
    "contrast": "low_contrast",
    "noise": "sensor_noise",
}


def candidate_label(row, severity_floor):
    # Impulse fraction is a separate signal, not general sensor-noise severity.
    if row["impulse_fraction"] >= IMPULSE_THRESHOLD:
        return "sensor_noise"

    blur = row["blur_severity"]
    low_contrast = row["low_contrast_severity"]

    # Test an explicit minimum-severity floor to avoid tiny residuals
    # being reported as a directional defect.
    if max(blur, low_contrast) <= severity_floor:
        return "none"

    # Stable tie priority: blur, then low contrast.
    return "motion_blur" if blur >= low_contrast else "low_contrast"


def evaluate(label, filename, image, amount=None):
    m2 = analyze_image_fuzzy(image)
    v2 = build_v2_payload(m2, image)
    impulse = estimate_impulse_noise(image)

    return {
        "class": label,
        "filename": filename,
        "injected_noise_amount": amount,
        "source_score": m2["fuzzy_score"],
        "source_state": m2["state"],
        "current_v2_defect": v2["dominant_defect"],
        "v2_state": v2["state"],
        "raw_blur": float(v2["metrics_raw"]["blur"]),
        "raw_contrast": float(v2["metrics_raw"]["contrast"]),
        "raw_linear_snr": (
            None if v2["metrics_raw"]["noise"] is None
            else float(v2["metrics_raw"]["noise"])
        ),
        "blur_severity": float(v2["metrics"]["blur"]),
        "low_contrast_severity": float(
            contrast_severity_parts(
                float(v2["metrics_raw"]["contrast"])
            )[1]
        ),
        "impulse_fraction": float(impulse["impulse_fraction"]),
        "impulse_percent": float(impulse["impulse_percent"]),
    }


def counts_by_class(rows, field):
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row["class"]][row[field]] += 1
    return {key: dict(value) for key, value in sorted(grouped.items())}


def main():
    if not MANIFEST.exists() or not SOURCE.exists():
        raise SystemExit("Missing fixture manifest or source DICOM.")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []

    clean = load_dicom(str(SOURCE))
    rows.append(evaluate(
        "clean_source",
        str(SOURCE.relative_to(ROOT)),
        clean,
    ))

    for record in manifest:
        path = DATA / record["filename"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Could not read fixture: {path}")

        amount = record.get("parameters", {}).get("amount")
        rows.append(evaluate(
            record["injected_defect"],
            str(path.relative_to(ROOT)),
            image,
            amount,
        ))

    corrupted = [row for row in rows if row["class"] in EXPECTED]
    noise = [row for row in corrupted if row["class"] == "noise"]
    other = [row for row in corrupted if row["class"] != "noise"]

    # Test how the impulse signal separates injected noise from the
    # other synthetic classes at the predeclared exploratory threshold.
    tp = sum(r["impulse_fraction"] >= IMPULSE_THRESHOLD for r in noise)
    fn = len(noise) - tp
    fp = sum(r["impulse_fraction"] >= IMPULSE_THRESHOLD for r in other)
    tn = len(other) - fp

    candidate_results = {}
    for floor in SEVERITY_FLOORS:
        field = f"candidate_floor_{floor:g}"
        for row in rows:
            row[field] = candidate_label(row, floor)

        predictions = [row[field] for row in corrupted]
        correct = sum(
            row[field] == EXPECTED[row["class"]]
            for row in corrupted
        )
        candidate_results[str(floor)] = {
            "severity_floor": floor,
            "correct_corrupted_fixtures": correct,
            "total_corrupted_fixtures": len(corrupted),
            "fixture_class_agreement": (
                correct / len(corrupted) if corrupted else None
            ),
            "predictions_by_class": counts_by_class(rows, field),
            "clean_reference_candidate": rows[0][field],
        }

    nonnoise_percent = [
        row["impulse_percent"]
        for row in rows if row["class"] in ("clean_source", "blur", "contrast")
    ]
    noise_percent = [row["impulse_percent"] for row in noise]

    report = {
        "status": "EXPLORATORY_NOT_CLINICALLY_VALIDATED",
        "total_images": len(rows),
        "corrupted_fixtures": len(corrupted),
        "distinct_augmented_source_images": len({
            record.get("source") for record in manifest
        }),
        "current_v2_predictions_by_class": counts_by_class(
            rows, "current_v2_defect"
        ),
        "current_v2_states_by_class": counts_by_class(rows, "v2_state"),
        "impulse_experiment": {
            "kernel_size": 3,
            "residual_threshold": 64.0,
            "classification_threshold_fraction": IMPULSE_THRESHOLD,
            "classification_threshold_percent": 100 * IMPULSE_THRESHOLD,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "min_noise_fixture_percent": min(noise_percent),
            "max_clean_blur_contrast_percent": max(nonnoise_percent),
            "clean_reference_percent": rows[0]["impulse_percent"],
        },
        "candidate_comparisons": candidate_results,
        "limitations": [
            "All degraded fixtures were generated from one source image.",
            "Class agreement measures synthetic fixture labels, not clinical accuracy.",
            "Impulse density targets salt-and-pepper corruption, not all sensor noise.",
            "Candidate labels and severity floors are exploratory, not production settings.",
            "The candidate experiment does not recalibrate the overall fuzzy quality score.",
        ],
        "reports": [
            str(OUT_JSON.relative_to(ROOT)),
            str(OUT_CSV.relative_to(ROOT)),
        ],
    }

    OUT_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    columns = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
