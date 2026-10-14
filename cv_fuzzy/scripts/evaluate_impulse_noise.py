
"""Exploratory comparison of SNR and impulse-noise estimators.

This script is evaluation-only. It does not modify production metrics,
fuzzy rules, state thresholds, or the v2 contract.

Run from the repository root:
    python scripts/evaluate_impulse_noise.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.fuzzy_engine import (
    BLUR_MAX,
    CONTRAST_MAX,
    SNR_MAX,
    _BLUR,
    _CONTRAST,
    _CONTROL_SYSTEM,
    _SNR,
    analyze_image_fuzzy,
)
from cv_fuzzy.impulse_noise import estimate_impulse_noise
from cv_fuzzy.v2_contract import (
    contrast_severity_parts,
    noise_severity,
    blur_severity,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "gold_standard"
MANIFEST = DATA_DIR / "manifest.json"
SOURCE = ROOT / "cv_fuzzy" / "sample_data" / "CT_small.dcm"
REPORTS = ROOT / "reports"

IMAGE_CSV = REPORTS / "impulse_noise_image_evaluation.csv"
SUMMARY_JSON = REPORTS / "impulse_noise_evaluation_summary.json"
THRESHOLD_CSV = REPORTS / "impulse_noise_threshold_sweep.csv"

# Baseline detection criterion:
# SNR < 2.5 is where the current SNR-low membership becomes positive.
BASELINE_SNR_THRESHOLD = 2.5

# Initial candidate decision threshold, selected before the threshold sweep.
# This is an exploratory engineering value, not a validated cutoff.
CANDIDATE_DETECTION_THRESHOLD = 0.005

# Experimental mapping used ONLY to compare counterfactual fuzzy outputs:
# a detected impulse fraction of 1% maps to 100 noise severity.
# This does not change the existing v2 noise_severity() function.


CLASS_ORDER = ["clean_source", "blur", "contrast", "noise"]


# Evaluation-only impulse-fraction fuzzy controller.
# This is a separate experimental model, not a production replacement.
IMPULSE_FRACTION_UNIVERSE_MAX = 1.0
IMPULSE_MEMBERSHIP_LOW_END = 0.0025
IMPULSE_DETECTION_THRESHOLD = CANDIDATE_DETECTION_THRESHOLD
IMPULSE_MEMBERSHIP_HIGH_FULL_AT = 0.0083


def _build_candidate_control_system():
    """Build a separate fuzzy controller using impulse fraction, not SNR."""

    # Copy the existing blur and contrast membership functions so their
    # definitions match the production model without sharing its controller.
    blur = ctrl.Antecedent(_BLUR.universe.copy(), "blur")
    for term in ("low", "medium", "high"):
        blur[term] = _BLUR[term].mf.copy()

    contrast = ctrl.Antecedent(
        _CONTRAST.universe.copy(), "contrast"
    )
    for term in ("low", "medium", "high"):
        contrast[term] = _CONTRAST[term].mf.copy()

    # Dense sampling near the observed synthetic impulse fractions.
    impulse_universe = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 0.05, 501),
                np.linspace(0.05, 1.0, 96),
            )
        )
    )
    impulse = ctrl.Antecedent(
        impulse_universe, "impulse_fraction"
    )

    impulse["low"] = fuzz.trapmf(
        impulse.universe,
        [0.0, 0.0, 0.0025, 0.005],
    )
    impulse["medium"] = fuzz.trimf(
        impulse.universe,
        [0.0025, 0.005, 0.0083],
    )
    impulse["high"] = fuzz.trapmf(
        impulse.universe,
        [0.005, 0.0083, 1.0, 1.0],
    )

    quality = ctrl.Consequent(
        np.linspace(0, 100, 101), "quality"
    )
    quality["poor"] = fuzz.trapmf(
        quality.universe, [0, 0, 20, 45]
    )
    quality["fair"] = fuzz.trimf(
        quality.universe, [25, 50, 75]
    )
    quality["good"] = fuzz.trapmf(
        quality.universe, [55, 80, 100, 100]
    )

    rules = [
        ctrl.Rule(blur["low"], quality["poor"]),
        ctrl.Rule(contrast["low"], quality["poor"]),
        ctrl.Rule(impulse["high"], quality["poor"]),
        ctrl.Rule(
            blur["medium"]
            & contrast["medium"]
            & impulse["medium"],
            quality["fair"],
        ),
        ctrl.Rule(
            blur["high"]
            & contrast["medium"]
            & impulse["low"],
            quality["good"],
        ),
        ctrl.Rule(
            blur["high"]
            & contrast["high"]
            & impulse["low"],
            quality["good"],
        ),
        ctrl.Rule(
            blur["medium"]
            & contrast["high"]
            & impulse["low"],
            quality["good"],
        ),
    ]

    return (
        ctrl.ControlSystem(rules),
        blur,
        contrast,
        impulse,
    )


(
    _CANDIDATE_CONTROL_SYSTEM,
    _CANDIDATE_BLUR,
    _CANDIDATE_CONTRAST,
    _CANDIDATE_IMPULSE,
) = _build_candidate_control_system()



def state_for_score(score: int) -> str:
    """Use the existing production state boundaries unchanged."""
    if score >= 75:
        return "ACCEPT"
    if score >= 45:
        return "BORDERLINE"
    return "REJECT"


def membership(variable, term: str, value: float) -> float:
    """Interpolate a term's membership at an input value."""
    return float(
        fuzz.interp_membership(
            variable.universe,
            variable[term].mf,
            value,
        )
    )


def fuzzy_inputs(metrics: dict, snr_override: float | None = None):
    """Apply the same input clipping used by analyze_image_fuzzy()."""
    blur_value = float(np.clip(metrics["blur"], 0, BLUR_MAX))
    contrast_value = float(
        np.clip(metrics["contrast"], 0, CONTRAST_MAX)
    )

    if snr_override is None:
        raw_snr = float(metrics["noise"])
        if math.isinf(raw_snr) and raw_snr > 0:
            snr_value = SNR_MAX
        else:
            snr_value = float(np.clip(raw_snr, 0, SNR_MAX))
            if not math.isfinite(snr_value):
                snr_value = 0.0
    else:
        snr_value = float(np.clip(snr_override, 0, SNR_MAX))

    return blur_value, contrast_value, snr_value



def defect_memberships(metrics: dict, snr_value: float) -> dict:
    """Reproduce the existing dominant-defect membership comparisons."""
    blur_value, contrast_value, snr_value = fuzzy_inputs(
        metrics, snr_override=snr_value
    )

    return {
        "motion_blur": membership(_BLUR, "low", blur_value),
        "low_contrast": membership(
            _CONTRAST, "low", contrast_value
        ),
        "high_noise": membership(_SNR, "low", snr_value),
    }


def dominant_defect(metrics: dict, snr_value: float) -> str:
    """Use the existing defect names and deterministic tie order."""
    values = defect_memberships(metrics, snr_value)
    selected = max(values, key=values.get)

    if values[selected] <= 0:
        return "none"

    return selected


def rule_activations(metrics: dict, snr_value: float) -> dict:
    """Record antecedent firing strengths for the seven existing rules.

    The AND operator uses the minimum of antecedent memberships.
    These are rule firing strengths, not normalized probabilities.
    """
    blur_value, contrast_value, snr_value = fuzzy_inputs(
        metrics, snr_override=snr_value
    )

    b = {
        term: membership(_BLUR, term, blur_value)
        for term in ("low", "medium", "high")
    }
    c = {
        term: membership(_CONTRAST, term, contrast_value)
        for term in ("low", "medium", "high")
    }
    s = {
        term: membership(_SNR, term, snr_value)
        for term in ("low", "medium", "high")
    }

    activations = {
        "r1_blur_low_to_poor": b["low"],
        "r2_contrast_low_to_poor": c["low"],
        "r3_snr_low_to_poor": s["low"],
        "r4_all_medium_to_fair": min(
            b["medium"], c["medium"], s["medium"]
        ),
        "r5_blur_high_contrast_medium_snr_high_to_good": min(
            b["high"], c["medium"], s["high"]
        ),
        "r6_blur_high_contrast_high_snr_high_to_good": min(
            b["high"], c["high"], s["high"]
        ),
        "r7_blur_medium_contrast_high_snr_high_to_good": min(
            b["medium"], c["high"], s["high"]
        ),
    }

    return {
        key: round(float(value), 8)
        for key, value in activations.items()
    }


def candidate_inputs(
    metrics: dict,
    impulse_fraction: float,
) -> tuple[float, float, float]:
    """Prepare inputs for the separate candidate controller."""
    blur_value = float(np.clip(metrics["blur"], 0, BLUR_MAX))
    contrast_value = float(
        np.clip(metrics["contrast"], 0, CONTRAST_MAX)
    )
    impulse_value = float(
        np.clip(impulse_fraction, 0, IMPULSE_FRACTION_UNIVERSE_MAX)
    )

    return blur_value, contrast_value, impulse_value


def simulate_candidate_fuzzy_score(
    metrics: dict,
    impulse_fraction: float,
) -> tuple[int, str]:
    """Score an image using the evaluation-only impulse controller."""
    blur_value, contrast_value, impulse_value = candidate_inputs(
        metrics, impulse_fraction
    )

    simulation = ctrl.ControlSystemSimulation(
        _CANDIDATE_CONTROL_SYSTEM
    )
    simulation.input["blur"] = blur_value
    simulation.input["contrast"] = contrast_value
    simulation.input["impulse_fraction"] = impulse_value
    simulation.compute()

    score = int(
        round(
            float(
                np.clip(
                    simulation.output["quality"], 0, 100
                )
            )
        )
    )
    return score, state_for_score(score)


def candidate_defect_memberships(
    metrics: dict,
    impulse_fraction: float,
) -> dict:
    """Return evaluation-only memberships, not production contract labels."""
    blur_value, contrast_value, impulse_value = candidate_inputs(
        metrics, impulse_fraction
    )

    return {
        "motion_blur": membership(
            _CANDIDATE_BLUR, "low", blur_value
        ),
        "low_contrast": membership(
            _CANDIDATE_CONTRAST, "low", contrast_value
        ),
        "impulse_noise": membership(
            _CANDIDATE_IMPULSE, "high", impulse_value
        ),
    }


def candidate_dominant_defect(
    metrics: dict,
    impulse_fraction: float,
) -> str:
    """Select the strongest candidate membership deterministically."""
    values = candidate_defect_memberships(metrics, impulse_fraction)
    selected = max(values, key=values.get)

    if values[selected] <= 0:
        return "none"

    return selected


def candidate_rule_activations(
    metrics: dict,
    impulse_fraction: float,
) -> dict:
    """Record antecedent firing strengths for the candidate rules."""
    blur_value, contrast_value, impulse_value = candidate_inputs(
        metrics, impulse_fraction
    )

    b = {
        term: membership(_CANDIDATE_BLUR, term, blur_value)
        for term in ("low", "medium", "high")
    }
    c = {
        term: membership(_CANDIDATE_CONTRAST, term, contrast_value)
        for term in ("low", "medium", "high")
    }
    n = {
        term: membership(_CANDIDATE_IMPULSE, term, impulse_value)
        for term in ("low", "medium", "high")
    }

    activations = {
        "r1_blur_low_to_poor": b["low"],
        "r2_contrast_low_to_poor": c["low"],
        "r3_impulse_high_to_poor": n["high"],
        "r4_all_medium_to_fair": min(
            b["medium"], c["medium"], n["medium"]
        ),
        "r5_blur_high_contrast_medium_impulse_low_to_good": min(
            b["high"], c["medium"], n["low"]
        ),
        "r6_blur_high_contrast_high_impulse_low_to_good": min(
            b["high"], c["high"], n["low"]
        ),
        "r7_blur_medium_contrast_high_impulse_low_to_good": min(
            b["medium"], c["high"], n["low"]
        ),
    }

    return {
        key: round(float(value), 8)
        for key, value in activations.items()
    }


def v2_noise_status_and_severity(image: np.ndarray) -> dict:
    """Describe the existing SNR raw-status representation."""
    pixels = image.astype(np.float64)
    mean_value = float(pixels.mean())
    std_value = float(pixels.std())

    if std_value == 0:
        if mean_value > 0:
            return {
                "status": "positive_infinity",
                "raw_json_value": None,
                "severity": 0.0,
            }

        # The current v2 contract rejects undefined 0/0 SNR.
        return {
            "status": "undefined_0_over_0_rejected",
            "raw_json_value": None,
            "severity": None,
        }

    raw_snr = mean_value / std_value

    return {
        "status": "finite",
        "raw_json_value": float(raw_snr),
        "severity": float(noise_severity(float(raw_snr))),
    }



def assess_image(
    label: str,
    filename: str,
    image: np.ndarray,
) -> dict:
    """Evaluate baseline and separate candidate models for one image."""
    baseline = analyze_image_fuzzy(image)
    metrics = baseline["metrics"]

    candidate = estimate_impulse_noise(
        image,
        kernel_size=3,
        residual_threshold=64.0,
    )
    fraction = float(candidate["impulse_fraction"])

    # Candidate score comes from its own controller. The fraction is
    # never converted into an equivalent SNR.
    candidate_score, candidate_state = simulate_candidate_fuzzy_score(
        metrics, fraction
    )

    baseline_snr_input = fuzzy_inputs(metrics)[2]
    baseline_rules = rule_activations(metrics, baseline_snr_input)
    candidate_rules = candidate_rule_activations(metrics, fraction)
    candidate_memberships = candidate_defect_memberships(
        metrics, fraction
    )
    v2_noise = v2_noise_status_and_severity(image)

    row = {
        "filename": filename,
        "label": label,
        "sample_count": 1,
        "blur_raw": float(metrics["blur"]),
        "contrast_raw": float(metrics["contrast"]),
        "snr_raw": float(metrics["noise"]),
        "v2_raw_noise_status": v2_noise["status"],
        "v2_raw_noise_json_value": v2_noise["raw_json_value"],
        "v2_noise_severity_baseline": v2_noise["severity"],
        "candidate_impulse_pixels": candidate["impulse_pixels"],
        "candidate_pixel_count": candidate["pixel_count"],
        "candidate_impulse_fraction": fraction,
        "candidate_impulse_percent": candidate["impulse_percent"],
        "candidate_impulse_fraction_status": "finite",
        "baseline_noise_detected": bool(
            float(metrics["noise"]) < BASELINE_SNR_THRESHOLD
        ),
        "candidate_noise_detected_fixed_threshold": bool(
            fraction >= CANDIDATE_DETECTION_THRESHOLD
        ),
        "baseline_fuzzy_score": baseline["fuzzy_score"],
        "baseline_state": baseline["state"],
        "baseline_dominant_defect": baseline["dominant_defect"],
        "candidate_controller": (
            "separate_experimental_impulse_fraction_controller"
        ),
        "candidate_controller_impulse_fraction_input": fraction,
        "candidate_experimental_fuzzy_score": candidate_score,
        "candidate_experimental_state": candidate_state,
        "candidate_experimental_dominant_defect":
            candidate_dominant_defect(metrics, fraction),
    }

    for name, value in baseline_rules.items():
        row[f"baseline_rule_{name}"] = value

    for name, value in candidate_rules.items():
        row[f"candidate_rule_{name}"] = value

    for name, value in candidate_memberships.items():
        row[f"candidate_membership_{name}"] = round(
            float(value), 8
        )

    return row



def binary_metrics(y_true: list[bool], predictions: list[bool]) -> dict:
    """Calculate standard binary noise-detection metrics."""
    tp = sum(t and p for t, p in zip(y_true, predictions))
    fp = sum((not t) and p for t, p in zip(y_true, predictions))
    fn = sum(t and (not p) for t, p in zip(y_true, predictions))
    tn = sum((not t) and (not p) for t, p in zip(y_true, predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "sample_count": len(y_true),
        "actual_noise_samples": sum(y_true),
        "actual_non_noise_samples": len(y_true) - sum(y_true),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
    }


def build_threshold_sweep(
    fractions: list[float],
    y_true: list[bool],
) -> list[dict]:
    """Evaluate the full range of observed candidate densities."""
    thresholds = {0.0, CANDIDATE_DETECTION_THRESHOLD}
    thresholds.update(float(value) for value in fractions)

    maximum = max(fractions, default=0.0)
    thresholds.add(
        maximum + max(1e-12, abs(maximum) * 1e-9)
    )

    results = []

    for threshold in sorted(thresholds):
        predictions = [
            value >= threshold for value in fractions
        ]
        results.append({
            "threshold": float(threshold),
            **binary_metrics(y_true, predictions),
        })

    return results


def per_class_comparison(
    rows: list[dict],
    methods: dict[str, list[bool]],
) -> dict:
    """Show detections and false positives separately for each true class."""
    result = {}

    for label in CLASS_ORDER:
        indices = [
            i for i, row in enumerate(rows)
            if row["label"] == label
        ]
        group = {"sample_count": len(indices)}

        for method, predictions in methods.items():
            detected = sum(predictions[i] for i in indices)

            group[f"{method}_noise_detections"] = detected

            if label == "noise":
                group[f"{method}_true_positives"] = detected
                group[f"{method}_false_negatives"] = (
                    len(indices) - detected
                )
                group[f"{method}_recall"] = round(
                    detected / len(indices), 6
                ) if indices else 0.0
            else:
                group[f"{method}_false_positives"] = detected
                group[f"{method}_false_positive_rate"] = round(
                    detected / len(indices), 6
                ) if indices else 0.0

        result[label] = group

    return result


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    if not SOURCE.exists():
        raise SystemExit(f"Missing source DICOM: {SOURCE}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or len(manifest) != 100:
        raise SystemExit(
            "Expected exactly 100 labeled records in the manifest."
        )

    source_image = load_dicom(str(SOURCE))

    rows = [
        assess_image(
            "clean_source",
            str(SOURCE.relative_to(ROOT)),
            source_image,
        )
    ]

    for entry in manifest:
        image_path = DATA_DIR / entry["filename"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise SystemExit(
                f"Could not read dataset image: {image_path}"
            )

        rows.append(
            assess_image(
                entry["injected_defect"],
                entry["filename"],
                image,
            )
        )

    if len(rows) != 101:
        raise SystemExit(
            f"Expected 101 total images, assessed {len(rows)}."
        )

    y_true = [row["label"] == "noise" for row in rows]
    fractions = [
        float(row["candidate_impulse_fraction"]) for row in rows
    ]

    baseline_predictions = [
        bool(row["baseline_noise_detected"]) for row in rows
    ]
    candidate_fixed_predictions = [
        bool(row["candidate_noise_detected_fixed_threshold"])
        for row in rows
    ]

    threshold_sweep = build_threshold_sweep(fractions, y_true)

    # The best threshold is selected on these same 101 images.
    # This result is exploratory and optimistic, not a held-out estimate.
    best_threshold_result = max(
        threshold_sweep,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["threshold"],
        ),
    )
    best_threshold = float(best_threshold_result["threshold"])

    candidate_best_predictions = [
        value >= best_threshold for value in fractions
    ]

    for row, prediction in zip(rows, candidate_best_predictions):
        row["candidate_noise_detected_best_f1_threshold"] = bool(
            prediction
        )

    methods = {
        "baseline_snr": baseline_predictions,
        "candidate_fixed": candidate_fixed_predictions,
        "candidate_best_f1_in_sample": candidate_best_predictions,
    }

    overall_metrics = {
        "baseline_snr": {
            "decision_rule": f"raw SNR < {BASELINE_SNR_THRESHOLD}",
            **binary_metrics(y_true, baseline_predictions),
        },
        "candidate_fixed_threshold": {
            "decision_rule": (
                "impulse_fraction >= "
                f"{CANDIDATE_DETECTION_THRESHOLD}"
            ),
            **binary_metrics(y_true, candidate_fixed_predictions),
        },
        "candidate_best_f1_in_sample": {
            "decision_rule": (
                f"impulse_fraction >= {best_threshold:.12g}"
            ),
            **binary_metrics(y_true, candidate_best_predictions),
            "selection_note": (
                "Threshold selected and evaluated on the same 101 "
                "images; performance is optimistic and not held out."
            ),
        },
    }

    class_counts = {
        label: sum(row["label"] == label for row in rows)
        for label in CLASS_ORDER
    }

    false_positive_files = {}
    for method, predictions in methods.items():
        false_positive_files[method] = {
            label: [
                rows[i]["filename"]
                for i, prediction in enumerate(predictions)
                if prediction
                and rows[i]["label"] == label
                and label != "noise"
            ]
            for label in CLASS_ORDER
            if label != "noise"
        }

    raw_status_counts = {}
    for row in rows:
        status = row["v2_raw_noise_status"]
        raw_status_counts[status] = (
            raw_status_counts.get(status, 0) + 1
        )

    summary = {
        "status": "EXPLORATORY_NOT_CLINICALLY_VALIDATED",
        "dataset": {
            "total_images_including_clean_source": len(rows),
            "labeled_degradations": len(manifest),
            "class_counts": class_counts,
            "source_image": str(SOURCE.relative_to(ROOT)),
            "independent_ct_sources": 1,
        },

        "candidate_estimator": {
            "formula": (
                "impulse_fraction = count of extreme (0 or 255) "
                "pixels with absolute residual from a 3x3 median "
                "filter >= 64, divided by total pixel count"
            ),
            "kernel_size": 3,
            "residual_threshold": 64.0,
            "initial_fixed_detection_threshold":
                CANDIDATE_DETECTION_THRESHOLD,
            "rationale": (
                "Investigates isolated saturated impulses while "
                "requiring a large local residual. Performance on "
                "blur, contrast, and the clean source must be judged "
                "from measured false positives."
            ),
            "experimental_fuzzy_controller": {
                "input": "candidate_impulse_fraction",
                "input_range": [0.0, 1.0],
                "membership_functions": {
                    "low": "trapmf([0, 0, 0.0025, 0.005])",
                    "medium": "trimf([0.0025, 0.005, 0.0083])",
                    "high": "trapmf([0.005, 0.0083, 1, 1])",
                },
                "rules": [
                    "blur low -> poor",
                    "contrast low -> poor",
                    "impulse fraction high -> poor",
                    "blur medium AND contrast medium AND impulse medium -> fair",
                    "blur high AND contrast medium AND impulse low -> good",
                    "blur high AND contrast high AND impulse low -> good",
                    "blur medium AND contrast high AND impulse low -> good",
                ],
                "state_thresholds_unchanged": {
                    "ACCEPT": "score >= 75",
                    "BORDERLINE": "45 <= score < 75",
                    "REJECT": "score < 45",
                },
                "limitations": (
                    "Separate experimental controller; its scores are "
                    "not directly comparable as calibrated quality "
                    "scores with the baseline controller."
                ),
            },
        },

        "fuzzy_comparison": {
            "baseline": "Existing analyze_image_fuzzy() output.",
            "counterfactual": (
                "The candidate is evaluated using a separate fuzzy "
                "controller whose explicit noise input is impulse_fraction. "
                "The candidate fraction is not converted to SNR. This "
                "controller is experimental and is not production behavior."
            ),
            "comparison_caveat": (
                "Baseline and candidate scores come from different noise "
                "antecedents and rule systems; score differences alone "
                "do not demonstrate improved overall image quality."
            ),

            "state_thresholds_unchanged": {
                "ACCEPT": "score >= 75",
                "BORDERLINE": "45 <= score < 75",
                "REJECT": "score < 45",
            },
            "rule_activations": (
                "Per-rule antecedent firing strengths are included "
                "in the per-image CSV; they are not probabilities."
            ),
        },
        "noise_detection_metrics": overall_metrics,
        "per_class_detection_comparison": per_class_comparison(
            rows, methods
        ),
        "false_positive_filenames_by_true_class": false_positive_files,
        "candidate_threshold_sweep": {
            "best_threshold": best_threshold,
            "best_threshold_metrics": best_threshold_result,
            "selection_note": (
                "In-sample threshold sweep for exploratory analysis; "
                "use independent images for unbiased evaluation."
            ),
            "csv_path": str(THRESHOLD_CSV.relative_to(ROOT)),
        },
        "v2_calibration_and_raw_status": {
            "baseline_status_counts": raw_status_counts,
            "existing_raw_noise_semantics": (
                "metrics_raw.noise is a linear SNR-like ratio. "
                "Positive infinity is represented as JSON null with "
                "status positive_infinity; undefined 0/0 SNR is rejected."
            ),
            "candidate_raw_metric": (
                "impulse_fraction is a distinct finite ratio in [0,1]."
            ),
            "compatibility": (
                "The candidate fraction must not be substituted into "
                "metrics_raw.noise or passed to noise_severity() as if "
                "it were SNR. A reviewed contract change would need to "
                "give the candidate metric its own semantics, calibration, "
                "and raw-status representation."
            ),
            "production_contract_changed": False,
            "strict_validation_changed": False,
            "experimental_candidate_severity_is_not_approved": True,
        },
        "limitations": [
            "All 100 synthetic degradations originate from one CT source.",
            "These results are exploratory and do not establish "
            "generalization to independent clinical images.",
            "The best-F1 threshold is tuned and evaluated on the same "
            "dataset; it is not an unbiased estimate of future performance.",
            "The separate experimental impulse-fraction controller's "
            "membership functions and rules are provisional. Its scores "
            "are not calibrated equivalents of the baseline scores.",

        ],
    }


    # Additional review summaries requested by M1.
    # These summaries use the existing per-image rows and predictions;
    # they do not alter either estimator or fuzzy controller.

    def _count_values(values):
        counts = {}
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def _score_distribution(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {"count": 0}

        return {
            "count": int(values.size),
            "min": float(values.min()),
            "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "p75": float(np.percentile(values, 75)),
            "max": float(values.max()),
            "population_std": float(values.std(ddof=0)),
        }

    def _fuzzy_profile(
        selected_rows, score_key, state_key, defect_key
    ):
        return {
            "sample_count": len(selected_rows),
            "score_distribution": _score_distribution([
                row[score_key] for row in selected_rows
            ]),
            "state_counts": _count_values([
                row[state_key] for row in selected_rows
            ]),
            "dominant_defect_counts": _count_values([
                row[defect_key] for row in selected_rows
            ]),
        }

    def _confusion_matrix(actual, predicted):
        tp = sum(bool(a) and bool(p) for a, p in zip(actual, predicted))
        fp = sum(not bool(a) and bool(p) for a, p in zip(actual, predicted))
        fn = sum(bool(a) and not bool(p) for a, p in zip(actual, predicted))
        tn = sum(not bool(a) and not bool(p) for a, p in zip(actual, predicted))

        return {
            "positive_class": "noise",
            "negative_class": "non_noise",
            "sample_count": len(actual),
            "matrix": {
                "actual_non_noise": {
                    "predicted_non_noise": int(tn),
                    "predicted_noise": int(fp),
                },
                "actual_noise": {
                    "predicted_non_noise": int(fn),
                    "predicted_noise": int(tp),
                },
            },
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }

    # 1. Explicit confusion matrices for baseline and both candidate
    #    operating points. The best-F1 result is in-sample, not held out.
    summary["confusion_matrices"] = {
        method: _confusion_matrix(y_true, predictions)
        for method, predictions in methods.items()
    }

    # 2. Clear per-class sample counts, recall for the noise class,
    #    and false-positive rates for each non-noise class.
    per_class_metrics = {}

    for label in CLASS_ORDER:
        indices = [
            i for i, row in enumerate(rows)
            if row["label"] == label
        ]
        class_result = {"sample_count": len(indices)}

        for method, predictions in methods.items():
            detected = sum(bool(predictions[i]) for i in indices)

            if label == "noise":
                class_result[method] = {
                    "actual_noise_samples": len(indices),
                    "true_positives": int(detected),
                    "false_negatives": int(len(indices) - detected),
                    "recall": (
                        float(detected / len(indices))
                        if indices else 0.0
                    ),
                }
            else:
                class_result[method] = {
                    "actual_non_noise_samples": len(indices),
                    "false_positives": int(detected),
                    "true_negatives_within_class": int(
                        len(indices) - detected
                    ),
                    "false_positive_rate": (
                        float(detected / len(indices))
                        if indices else 0.0
                    ),
                }

        per_class_metrics[label] = class_result

    summary["per_class_detection_metrics"] = per_class_metrics

    # 3. Score distributions, state counts, and dominant-defect counts.
    baseline_keys = (
        "baseline_fuzzy_score",
        "baseline_state",
        "baseline_dominant_defect",
    )
    candidate_keys = (
        "candidate_experimental_fuzzy_score",
        "candidate_experimental_state",
        "candidate_experimental_dominant_defect",
    )

    summary["fuzzy_output_summaries"] = {
        "interpretation": (
            "Candidate scores come from a separate experimental "
            "controller. Score differences alone do not demonstrate "
            "improved overall image quality."
        ),
        "overall": {
            "baseline_production": _fuzzy_profile(
                rows, *baseline_keys
            ),
            "candidate_experimental": _fuzzy_profile(
                rows, *candidate_keys
            ),
        },
        "by_class": {},
    }

    for label in CLASS_ORDER:
        selected_rows = [
            row for row in rows if row["label"] == label
        ]

        summary["fuzzy_output_summaries"]["by_class"][label] = {
            "baseline_production": _fuzzy_profile(
                selected_rows, *baseline_keys
            ),
            "candidate_experimental": _fuzzy_profile(
                selected_rows, *candidate_keys
            ),
        }

    # 4. Activation summaries for all rules across all 101 images and
    #    separately within each labeled image class.
    def _activation_statistics(selected_rows, field):
        values = np.asarray(
            [float(row[field]) for row in selected_rows],
            dtype=np.float64,
        )

        if values.size == 0:
            return {"sample_count": 0}

        active_count = int(np.count_nonzero(values > 0.0))

        return {
            "sample_count": int(values.size),
            "active_count": active_count,
            "zero_count": int(values.size - active_count),
            "active_fraction": float(active_count / values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "max": float(values.max()),
        }

    rule_groups = {
        "baseline_production": [
            field for field in rows[0]
            if field.startswith("baseline_rule_")
        ],
        "candidate_experimental": [
            field for field in rows[0]
            if field.startswith("candidate_rule_")
        ],
    }

    summary["rule_activation_statistics"] = {
        "interpretation": (
            "Membership-based antecedent firing strengths, not "
            "probabilities. Every per-image value remains available "
            "in the image evaluation CSV."
        ),
        "overall": {},
        "by_class": {},
    }

    for model_name, fields in rule_groups.items():
        summary["rule_activation_statistics"]["overall"][
            model_name
        ] = {
            field: _activation_statistics(rows, field)
            for field in sorted(fields)
        }

    for label in CLASS_ORDER:
        selected_rows = [
            row for row in rows if row["label"] == label
        ]

        summary["rule_activation_statistics"]["by_class"][label] = {}

        for model_name, fields in rule_groups.items():
            summary["rule_activation_statistics"]["by_class"][label][
                model_name
            ] = {
                field: _activation_statistics(selected_rows, field)
                for field in sorted(fields)
            }

    # 5. Document the origin of each threshold, without implying that
    #    the exploratory in-sample optimum is independently validated.
    summary["candidate_estimator"]["threshold_selection"] = {
        "fixed_threshold": {
            "impulse_fraction": float(
                CANDIDATE_DETECTION_THRESHOLD
            ),
            "percent": float(
                100.0 * CANDIDATE_DETECTION_THRESHOLD
            ),
            "selection_basis": (
                "Initial engineering cutoff specified for the "
                "exploratory evaluation; not selected by maximizing "
                "the threshold-sweep F1."
            ),
            "optimized_on_this_dataset": False,
        },
        "best_f1_in_sample": {
            "impulse_fraction": float(best_threshold),
            "percent": float(100.0 * best_threshold),
            "selection_objective": (
                "Maximize F1; ties are resolved by precision and "
                "then the larger threshold."
            ),
            "selected_and_evaluated_on_same_101_images": True,
            "held_out_performance_estimate": False,
        },
    }


    # Direct case comparison requested by M1.
    # Preserve individual rule activations alongside each example.
    def _case_snapshot(row):
        return {
            "filename": row["filename"],
            "label": row["label"],
            "impulse_fraction": float(
                row["candidate_impulse_fraction"]
            ),
            "candidate_noise_detection": {
                "baseline_snr_detected": bool(
                    row["baseline_noise_detected"]
                ),
                "candidate_fixed_threshold_detected": bool(
                    row["candidate_noise_detected_fixed_threshold"]
                ),
            },
            "baseline": {
                "fuzzy_score": int(row["baseline_fuzzy_score"]),
                "state": row["baseline_state"],
                "dominant_defect": row["baseline_dominant_defect"],
                "rule_activations": {
                    key: float(value)
                    for key, value in row.items()
                    if key.startswith("baseline_rule_")
                },
            },
            "candidate_experimental": {
                "fuzzy_score": int(
                    row["candidate_experimental_fuzzy_score"]
                ),
                "state": row["candidate_experimental_state"],
                "dominant_defect": row[
                    "candidate_experimental_dominant_defect"
                ],
                "rule_activations": {
                    key: float(value)
                    for key, value in row.items()
                    if key.startswith("candidate_rule_")
                },
            },
        }

    representative_cases = {}

    for label in CLASS_ORDER:
        candidates_for_label = [
            row for row in rows if row["label"] == label
        ]

        if not candidates_for_label:
            continue

        if label == "noise":
            # Show the least dense labeled noise example.
            chosen = min(
                candidates_for_label,
                key=lambda row: float(
                    row["candidate_impulse_fraction"]
                ),
            )
        else:
            chosen = candidates_for_label[0]

        representative_cases[label] = _case_snapshot(chosen)

    # Verify the clean-source outcome and count candidate impulse-noise
    # dominant labels on non-noise classes without assuming correctness.
    clean_source_rows = [
        row for row in rows if row["label"] == "clean_source"
    ]
    clean_result = (
        _case_snapshot(clean_source_rows[0])
        if clean_source_rows else None
    )

    non_noise_rows = [
        row for row in rows if row["label"] != "noise"
    ]

    candidate_impulse_defects_on_non_noise = [
        row for row in non_noise_rows
        if row["candidate_experimental_dominant_defect"]
        == "impulse_noise"
    ]

    summary["direct_case_comparison"] = {
        "clean_source": clean_result,
        "representative_examples_by_class": representative_cases,
        "candidate_impulse_noise_dominant_defect_on_non_noise": {
            "sample_count": len(non_noise_rows),
            "false_impulse_defect_count": len(
                candidate_impulse_defects_on_non_noise
            ),
            "filenames": [
                row["filename"]
                for row in candidate_impulse_defects_on_non_noise
            ],
        },
        "observed_baseline_noise_false_positives_by_class": {
            label: sum(
                bool(row["baseline_noise_detected"])
                for row in rows
                if row["label"] == label
            )
            for label in CLASS_ORDER
            if label != "noise"
        },
        "interpretation": (
            "The clean source's baseline rejection and experimental "
            "candidate outcome are reported explicitly. An ACCEPT/none "
            "candidate result is an observation about the separate "
            "experimental controller, not a production-quality guarantee. "
            "Candidate dominant defects on non-noise examples are counted "
            "from measured outputs rather than assumed."
        ),
        "dataset_observation": (
            "For this dataset and raw-SNR < 2.5 baseline detection rule, "
            "the clean source and blur examples are noise false positives; "
            "the contrast class has zero binary noise false positives. "
            "Contrast examples may still have low_contrast as their fuzzy "
            "dominant defect."
        ),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)

    with IMAGE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with THRESHOLD_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(threshold_sweep[0].keys())
        )
        writer.writeheader()
        writer.writerows(threshold_sweep)

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(json.dumps({
        "status": summary["status"],
        "image_count": len(rows),
        "class_counts": class_counts,
        "noise_detection_metrics": overall_metrics,
        "per_class_detection_comparison":
            summary["per_class_detection_comparison"],
        "v2_raw_status_counts": raw_status_counts,
        "artifacts": [
            str(IMAGE_CSV.relative_to(ROOT)),
            str(SUMMARY_JSON.relative_to(ROOT)),
            str(THRESHOLD_CSV.relative_to(ROOT)),
        ],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
