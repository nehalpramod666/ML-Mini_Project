"""Audit provisional M2 v2 calibrations on data/gold_standard (read-only).

Run from repository root in the project's venv:
  python scripts/audit_gold_standard_v2.py

Writes reports/gold_standard_v2_audit.json and .csv. These generated evaluation
results must be reviewed before any anchor is described as validated.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.metrics import compute_blur, compute_contrast, compute_noise
from cv_fuzzy.v2_contract import blur_severity, contrast_severity_parts, noise_severity

ROOT = Path.cwd()
DATA = ROOT / "data" / "gold_standard"
MANIFEST = DATA / "manifest.json"
SOURCE = ROOT / "cv_fuzzy" / "sample_data" / "CT_small.dcm"
REPORTS = ROOT / "reports"


def summary(values):
    vals = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
    if vals.size == 0:
        return {"n_finite": 0}
    return {
        "n_finite": int(vals.size),
        "min": round(float(np.min(vals)), 6),
        "q1": round(float(np.percentile(vals, 25)), 6),
        "median": round(float(np.median(vals)), 6),
        "q3": round(float(np.percentile(vals, 75)), 6),
        "max": round(float(np.max(vals)), 6),
    }


def lower_snr_auc(noise_values, comparator_values):
    """Pairwise AUC where lower SNR predicts the `noise` label; ties count 0.5."""
    a = [float(x) for x in noise_values if x is not None and math.isfinite(float(x))]
    b = [float(x) for x in comparator_values if x is not None and math.isfinite(float(x))]
    if not a or not b:
        return None
    wins = 0.0
    total = len(a) * len(b)
    for noise_snr in a:
        for other_snr in b:
            if noise_snr < other_snr:
                wins += 1.0
            elif noise_snr == other_snr:
                wins += 0.5
    return round(wins / total, 6)


def read_gray(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read grayscale image: {path}")
    return image


def assess(label, filename, image):
    x = image.astype(np.float64)
    mean, std = float(np.mean(x)), float(np.std(x))
    raw_blur = float(compute_blur(image))
    raw_contrast = float(compute_contrast(image))
    raw_noise = float(compute_noise(image))

    if std == 0.0:
        if mean > 0.0:
            snr_status, snr = "positive_infinity", None
            n_sev = 0.0
        elif mean == 0.0:
            snr_status, snr = "undefined_zero_over_zero", None
            n_sev = None
        else:
            snr_status, snr, n_sev = "invalid", None, None
    else:
        if not math.isfinite(raw_noise) or raw_noise < 0:
            snr_status, snr, n_sev = "invalid", None, None
        else:
            snr_status, snr = "finite", raw_noise
            n_sev = noise_severity(raw_noise)

    c_sev, c_low, c_high = contrast_severity_parts(raw_contrast)
    record = {
        "label": label,
        "filename": filename,
        "raw_blur_variance": raw_blur,
        "raw_contrast_std_gray_levels": raw_contrast,
        "raw_noise_linear_snr": snr,
        "raw_noise_status": snr_status,
        "blur_severity_candidate": blur_severity(raw_blur),
        "contrast_severity_candidate": c_sev,
        "low_contrast_severity_candidate": c_low,
        "high_contrast_severity_candidate_disabled": c_high,
        "noise_severity_candidate": n_sev,
        "pixel_mean": mean,
        "pixel_std": std,
    }
    return record


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    if not SOURCE.exists():
        raise SystemExit(f"Missing clean source DICOM: {SOURCE}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = [assess("clean_source", str(SOURCE.relative_to(ROOT)), load_dicom(str(SOURCE)))]
    errors = []
    for entry in manifest:
        try:
            rows.append(assess(
                entry["injected_defect"],
                entry["filename"],
                read_gray(DATA / entry["filename"]),
            ))
        except Exception as exc:
            errors.append({"filename": entry.get("filename"), "error": str(exc)})

    classes = {}
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        classes[label] = {
            "count": len(group),
            "snr_linear": summary([r["raw_noise_linear_snr"] for r in group]),
            "noise_severity_candidate": summary([r["noise_severity_candidate"] for r in group]),
            "blur_variance": summary([r["raw_blur_variance"] for r in group]),
            "contrast_std": summary([r["raw_contrast_std_gray_levels"] for r in group]),
            "undefined_or_invalid_snr_count": sum(
                r["raw_noise_status"] in {"undefined_zero_over_zero", "invalid"} for r in group
            ),
        }

    noise = [r["raw_noise_linear_snr"] for r in rows if r["label"] == "noise"]
    other_degraded = [r["raw_noise_linear_snr"] for r in rows if r["label"] in {"blur", "contrast"}]
    all_other = [r["raw_noise_linear_snr"] for r in rows if r["label"] != "noise"]
    clean = next((r for r in rows if r["label"] == "clean_source"), None)
    clean_snr = clean["raw_noise_linear_snr"] if clean else None
    finite_noise = [x for x in noise if x is not None and math.isfinite(float(x))]

    report = {
        "status": "PROVISIONAL_NOT_VALIDATED",
        "dataset_manifest_records": len(manifest),
        "successfully_audited_images_including_clean_source": len(rows),
        "errors": errors,
        "classes": classes,
        "noise_separation": {
            "auc_noise_vs_blur_and_contrast_lower_snr_means_noise": lower_snr_auc(noise, other_degraded),
            "auc_noise_vs_all_non_noise_lower_snr_means_noise": lower_snr_auc(noise, all_other),
            "clean_source_snr": clean_snr,
            "noise_examples_with_snr_below_clean_source": (
                sum(float(x) < float(clean_snr) for x in finite_noise)
                if clean_snr is not None and math.isfinite(float(clean_snr)) else None
            ),
            "finite_noise_example_count": len(finite_noise),
            "interpretation": "Exploratory synthetic-data discrimination only; not clinical validation.",
        },
        "provisional_formulas": {
            "blur": "100*clip(1-log1p(variance)/log1p(110),0,1); cap 110 provisional",
            "noise": "100*clip(1-linear_snr/10,0,1); SNR dimensionless, not dB",
            "contrast_low": "100*clip((38-contrast_std)/18,0,1); anchors provisional",
            "contrast_high": "100*clip((contrast_std-60)/20,0,1); trigger disabled; anchors provisional",
        },
        "rows": rows,
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "gold_standard_v2_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with (REPORTS / "gold_standard_v2_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({k: report[k] for k in (
        "status", "dataset_manifest_records", "successfully_audited_images_including_clean_source",
        "errors", "classes", "noise_separation"
    )}, indent=2, allow_nan=False))
    print(f"\nFull reports written to:\n  {REPORTS / 'gold_standard_v2_audit.json'}\n  {REPORTS / 'gold_standard_v2_audit.csv'}")
    if errors:
        raise SystemExit("Audit completed with image errors; do not freeze calibration.")


if __name__ == "__main__":
    main()
