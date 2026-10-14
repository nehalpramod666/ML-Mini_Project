
"""Read-only exploratory analysis of M2 v2 score and defect semantics.

Run from the repository root:
    python scripts/evaluate_v2_semantics.py

Production contract, adapter, fuzzy rules, and thresholds are unchanged.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cv_fuzzy.v2_contract import (
    blur_severity,
    contrast_severity_parts,
    noise_severity,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "m2_v2_fixture_generation.json"
OUTPUT_JSON = ROOT / "reports" / "v2_semantics_evaluation.json"
OUTPUT_CSV = ROOT / "reports" / "v2_semantics_per_image.csv"

CLASS_ORDER = ["clean_source", "blur", "contrast", "noise"]
SNR_REFERENCE_MAX = 10.0


def finite_number(value, field: str) -> float:
    """Parse a finite numeric fixture value and reject invalid inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not a numeric value: {value!r}")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def distribution(values) -> dict:
    """Summarize a non-empty list of finite numerical values."""
    data = np.asarray(list(values), dtype=np.float64)

    if data.size == 0:
        return {"count": 0}

    if not np.isfinite(data).all():
        raise ValueError("Cannot summarize non-finite values")

    return {
        "count": int(data.size),
        "min": float(data.min()),
        "p25": float(np.percentile(data, 25)),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "p75": float(np.percentile(data, 75)),
        "max": float(data.max()),
    }


def counts(values) -> dict:
    return dict(sorted(Counter(str(value) for value in values).items()))


def log_snr_severity(snr: float) -> float:
    """Illustrative alternative curve; not a proposed production formula."""
    return float(
        np.clip(
            100.0
            * (1.0 - math.log1p(snr) / math.log1p(SNR_REFERENCE_MAX)),
            0.0,
            100.0,
        )
    )


def clean_reference_delta_severity(
    snr: float,
    clean_reference_snr: float,
) -> float:
    """Source-specific counterfactual, intentionally not generalizable."""
    if clean_reference_snr <= 0:
        raise ValueError("Clean-reference SNR must be positive")

    return float(
        np.clip(
            100.0
            * (clean_reference_snr - snr)
            / clean_reference_snr,
            0.0,
            100.0,
        )
    )


def choose_defect(
    blur_value: float,
    low_contrast_value: float,
    noise_value: float,
) -> str:
    """Mirror the current enabled v2 labels and tie priority.

    High contrast remains disabled, as it is in the current contract.
    """
    candidates = [
        (blur_value, "motion_blur"),
        (low_contrast_value, "low_contrast"),
        (noise_value, "sensor_noise"),
    ]

    maximum = max(value for value, _ in candidates)
    if maximum == 0.0:
        return "none"

    # The order above deliberately preserves the current tie priority.
    return next(
        label for value, label in candidates if value == maximum
    )


def main() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing fixture report: {INPUT}")

    fixture_report = json.loads(INPUT.read_text(encoding="utf-8"))
    fixtures = fixture_report.get("fixtures")

    if not isinstance(fixtures, list) or len(fixtures) != 101:
        raise SystemExit(
            "Expected 101 fixtures (clean source plus 100 degradations)"
        )

    rows = []
    grouped = defaultdict(list)

    for item in fixtures:
        raw = item["raw_m2_output"]
        payload = item.get("v2_output") or {}
        metrics = raw["metrics"]

        label = item["label"]
        blur_raw = finite_number(metrics["blur"], "metrics.blur")
        contrast_raw = finite_number(metrics["contrast"], "metrics.contrast")
        snr_raw = finite_number(metrics["noise"], "metrics.noise")

        blur_s = blur_severity(blur_raw)
        contrast_s, low_contrast_s, high_contrast_s = (
            contrast_severity_parts(contrast_raw)
        )
        base_noise_s = noise_severity(snr_raw)

        # SNR = mean intensity / standard deviation.
        # contrast_raw is the standard deviation, so their product gives
        # the mean intensity implied by the existing metric definitions.
        implied_mean = snr_raw * contrast_raw

        row = {
            "filename": item["filename"],
            "class": label,
            "fixture_status": item["status"],
            "source_m2_score": int(raw["fuzzy_score"]),
            "source_m2_state": raw["state"],
            "source_m2_defect": raw["dominant_defect"],
            "v2_score": payload.get("fuzzy_score"),
            "v2_state": payload.get("state"),
            "v2_defect": payload.get("dominant_defect"),
            "raw_blur": blur_raw,
            "raw_contrast": contrast_raw,
            "raw_linear_snr": snr_raw,
            "implied_mean_intensity": implied_mean,
            "blur_severity": blur_s,
            "contrast_severity": contrast_s,
            "low_contrast_severity": low_contrast_s,
            "high_contrast_severity": high_contrast_s,
            "existing_linear_snr_noise_severity": base_noise_s,
            "log_curve_noise_severity": log_snr_severity(snr_raw),
            "source_reference_delta_noise_severity": None,
        }

        rows.append(row)
        grouped[label].append(row)

    if "clean_source" not in grouped or len(grouped["clean_source"]) != 1:
        raise SystemExit("Expected exactly one clean-source reference")

    clean_reference_snr = grouped["clean_source"][0]["raw_linear_snr"]

    for row in rows:
        row["source_reference_delta_noise_severity"] = (
            clean_reference_delta_severity(
                row["raw_linear_snr"], clean_reference_snr
            )
        )

        row["winner_existing"] = choose_defect(
            row["blur_severity"],
            row["low_contrast_severity"],
            row["existing_linear_snr_noise_severity"],
        )
        row["winner_log_curve"] = choose_defect(
            row["blur_severity"],
            row["low_contrast_severity"],
            row["log_curve_noise_severity"],
        )
        row["winner_source_reference_delta"] = choose_defect(
            row["blur_severity"],
            row["low_contrast_severity"],
            row["source_reference_delta_noise_severity"],
        )

    class_summary = {}

    for label in CLASS_ORDER:
        records = grouped[label]

        class_summary[label] = {
            "sample_count": len(records),
            "fixture_status_counts": counts(
                r["fixture_status"] for r in records
            ),
            "source_m2": {
                "score_distribution": distribution(
                    r["source_m2_score"] for r in records
                ),
                "state_counts": counts(
                    r["source_m2_state"] for r in records
                ),
                "dominant_defect_counts": counts(
                    r["source_m2_defect"] for r in records
                ),
            },
            "v2": {
                "score_distribution": distribution(
                    r["v2_score"] for r in records
                ),
                "state_counts": counts(
                    r["v2_state"] for r in records
                ),
                "dominant_defect_counts": counts(
                    r["v2_defect"] for r in records
                ),
            },
            "raw_linear_snr": distribution(
                r["raw_linear_snr"] for r in records
            ),
            "implied_mean_intensity": distribution(
                r["implied_mean_intensity"] for r in records
            ),
            "raw_contrast_std": distribution(
                r["raw_contrast"] for r in records
            ),
            "severity_distributions": {
                key: distribution(r[key] for r in records)
                for key in (
                    "blur_severity",
                    "low_contrast_severity",
                    "existing_linear_snr_noise_severity",
                    "log_curve_noise_severity",
                    "source_reference_delta_noise_severity",
                )
            },
            "count_noise_severity_exceeds_blur": sum(
                r["existing_linear_snr_noise_severity"]
                > r["blur_severity"]
                for r in records
            ),
            "count_noise_severity_exceeds_low_contrast": sum(
                r["existing_linear_snr_noise_severity"]
                > r["low_contrast_severity"]
                for r in records
            ),
            "counterfactual_dominant_defect_counts": {
                "existing_linear_snr": counts(
                    r["winner_existing"] for r in records
                ),
                "log_curve": counts(
                    r["winner_log_curve"] for r in records
                ),
                "clean_source_reference_delta": counts(
                    r["winner_source_reference_delta"] for r in records
                ),
            },
        }

    # Exploratory detection metrics: compare the selected defect with
    # the manifest's synthetic noise class. This is not clinical validation.
    def defect_detection_profile(winner_key):
        actual_noise = [
            row["class"] == "noise" for row in rows
        ]
        predicted_noise = [
            row[winner_key] == "sensor_noise" for row in rows
        ]

        tp = sum(
            actual and predicted
            for actual, predicted in zip(actual_noise, predicted_noise)
        )
        fp = sum(
            not actual and predicted
            for actual, predicted in zip(actual_noise, predicted_noise)
        )
        fn = sum(
            actual and not predicted
            for actual, predicted in zip(actual_noise, predicted_noise)
        )
        tn = sum(
            not actual and not predicted
            for actual, predicted in zip(actual_noise, predicted_noise)
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        per_class = {}

        for label in CLASS_ORDER:
            subset = [
                row for row in rows if row["class"] == label
            ]
            predicted_count = sum(
                row[winner_key] == "sensor_noise"
                for row in subset
            )

            entry = {
                "sample_count": len(subset),
                "sensor_noise_predictions": int(predicted_count),
                "sensor_noise_prediction_rate": (
                    float(predicted_count / len(subset))
                    if subset else 0.0
                ),
            }

            if label == "noise":
                entry.update({
                    "true_positives": int(predicted_count),
                    "false_negatives": int(
                        len(subset) - predicted_count
                    ),
                    "recall": (
                        float(predicted_count / len(subset))
                        if subset else 0.0
                    ),
                })
            else:
                entry.update({
                    "false_positives": int(predicted_count),
                    "true_negatives_within_class": int(
                        len(subset) - predicted_count
                    ),
                })

            per_class[label] = entry

        return {
            "positive_class": "noise",
            "prediction_rule": (
                f"{winner_key} selects sensor_noise"
            ),
            "sample_count": len(rows),
            "confusion_matrix": {
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
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "per_class": per_class,
        }

    defect_detection_metrics = {
        "interpretation": (
            "Exploratory agreement between the selected sensor_noise "
            "label and the synthetic manifest's noise class. These "
            "metrics do not establish clinical validity or causation."
        ),
        "existing_linear_snr": defect_detection_profile(
            "winner_existing"
        ),
        "log_curve": defect_detection_profile(
            "winner_log_curve"
        ),
        "clean_source_reference_delta": defect_detection_profile(
            "winner_source_reference_delta"
        ),
    }

    report = {
        "status": "EXPLORATORY_NOT_CLINICALLY_VALIDATED",
        "input_fixture_report": str(INPUT.relative_to(ROOT)),
        "input_fixture_status": fixture_report["status"],
        "input_fixture_success_count": fixture_report["success_count"],
        "input_fixture_blocked_count": fixture_report["blocked_count"],
        "total_samples": len(rows),
        "class_counts": {
            label: len(grouped[label]) for label in CLASS_ORDER
        },
        "current_contract": {
            "state_is_derived_from_fuzzy_score": True,
            "accept_threshold": 75,
            "borderline_lower_threshold": 45,
            "reject_threshold": 45,
            "noise_formula": (
                "100 * clip(1 - linear_snr / 10, 0, 1)"
            ),
            "defect_tie_priority": [
                "motion_blur",
                "low_contrast",
                "sensor_noise",
            ],
            "high_contrast_trigger_enabled": False,
        },
        "alternative_formula_comparisons": {
            "log_curve": {
                "formula": (
                    "100 * clip(1 - log1p(linear_snr) / log1p(10), 0, 1)"
                ),
                "selection_basis": (
                    "Illustrative curve-shape sensitivity analysis; "
                    "not tuned or validated as a replacement."
                ),
                "limitation": (
                    "A monotonic transformation changes severity scale "
                    "but does not fix the fact that image mean and image "
                    "standard deviation both influence the raw SNR proxy."
                ),
            },
            "clean_source_reference_delta": {
                "formula": (
                    "100 * clip((clean_reference_snr - snr) / "
                    "clean_reference_snr, 0, 1)"
                ),
                "reference_snr": clean_reference_snr,
                "selection_basis": (
                    "Uses the one clean image in this dataset as an "
                    "explicit reference; this is source-specific and "
                    "data-dependent."
                ),
                "limitation": (
                    "Counterfactual only. It must not be treated as a "
                    "validated general-purpose calibration or serialized "
                    "as a successful v2 contract output."
                ),
            },
        },
        "classes": class_summary,
        "defect_detection_metrics": defect_detection_metrics,
        "interpretation": [
            (
                "State is copied from the source fuzzy score and derived "
                "from the score thresholds; changing noise severity alone "
                "cannot repair the all-REJECT state distribution."
            ),
            (
                "Linear SNR is mean intensity divided by image standard "
                "deviation. Because standard deviation is also the raw "
                "contrast metric, this ratio is affected by global "
                "intensity spread and image structure, not only impulse noise."
            ),
            (
                "The log-curve and clean-reference alternatives are "
                "offline sensitivity analyses, not recommended production "
                "changes. The clean-reference alternative uses one source "
                "and must not be interpreted as independent validation."
            ),
            (
                "All 100 synthetic degradations originate from one CT "
                "source. Results do not establish generalization to "
                "independent clinical images."
            ),
        ],
        "production_changes": {
            "v2_contract_changed": False,
            "fixture_generator_changed": False,
            "m1_adapter_changed": False,
            "strict_validation_changed": False,
        },
        "per_image_csv": str(OUTPUT_CSV.relative_to(ROOT)),
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    import csv

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": report["status"],
        "total_samples": report["total_samples"],
        "class_counts": report["class_counts"],
        "source_m2_states_by_class": {
            label: class_summary[label]["source_m2"]["state_counts"]
            for label in CLASS_ORDER
        },
        "v2_defects_by_class": {
            label: class_summary[label]["v2"]["dominant_defect_counts"]
            for label in CLASS_ORDER
        },
        "counterfactual_winners_by_class": {
            label: class_summary[label][
                "counterfactual_dominant_defect_counts"
            ]
            for label in CLASS_ORDER
        },
        "reports": [
            str(OUTPUT_JSON.relative_to(ROOT)),
            str(OUTPUT_CSV.relative_to(ROOT)),
        ],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
