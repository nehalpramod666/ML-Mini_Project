"""Exploratory analysis of source M2 scores versus v2 defect selection.

Reads the existing fixture report and writes new diagnostic artifacts only.
Does not change production behavior or calibration thresholds.

This is single-source synthetic evaluation, not clinical validation.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path.cwd()
INPUT = ROOT / "reports" / "m2_v2_fixture_generation.json"
OUT_JSON = ROOT / "reports" / "v2_calibration_diagnostic_summary.json"
OUT_CSV = ROOT / "reports" / "v2_calibration_diagnostic_per_image.csv"

# Candidate mappings are used only to produce a transparent exploratory tally.
# In particular, "noise" -> "sensor_noise" is NOT an approved semantic mapping.
SOURCE_EXPECTED_LABEL = {
    "clean_source": "none",
    "blur": "motion_blur",
    "contrast": "low_contrast",
    "noise": "high_noise",
}
V2_EXPECTED_LABEL = {
    "clean_source": "none",
    "blur": "motion_blur",
    "contrast": "low_contrast",
    "noise": "sensor_noise",
}

def number(value):
    """Return a finite numeric value, otherwise None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def percentile(values, p):
    """Linearly interpolated percentile without third-party dependencies."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (
        position - lower
    )


def describe(values):
    values = [number(v) for v in values]
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "min": None, "p25": None, "median": None,
                "p75": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "max": max(values),
    }


def counts(values):
    return dict(sorted(Counter(
        str(v) if v is not None else "missing" for v in values
    ).items()))


def one_vs_rest(rows, reference_class, predicted_label):
    actual_positive = [
        r for r in rows if r["reference_class"] == reference_class
    ]
    tp = sum(r["predicted_label"] == predicted_label for r in actual_positive)
    fn = len(actual_positive) - tp
    other_rows = [r for r in rows if r["reference_class"] != reference_class]
    fp = sum(r["predicted_label"] == predicted_label for r in other_rows)
    tn = len(other_rows) - fp

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None
        and precision + recall > 0
        else None
    )
    return {
        "reference_class": reference_class,
        "candidate_positive_label": predicted_label,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing input report: {INPUT}")

    report = json.loads(INPUT.read_text(encoding="utf-8"))
    fixtures = report.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise SystemExit("Input report has no non-empty fixtures list.")

    rows = []
    for item in fixtures:
        raw_result = item.get("raw_m2_output") or {}
        raw_metrics = raw_result.get("metrics") or {}
        v2 = item.get("v2_output") or {}
        v2_raw = v2.get("metrics_raw") or {}
        v2_raw_status = v2.get("metrics_raw_status") or {}
        v2_metrics = v2.get("metrics") or {}

        source_score = number(raw_result.get("fuzzy_score"))
        v2_score = number(v2.get("fuzzy_score"))
        blur_sev = number(v2_metrics.get("blur"))
        noise_sev = number(v2_metrics.get("noise"))

        row = {
            "reference_class": item.get("label"),
            "filename": item.get("filename"),
            "fixture_status": item.get("status"),
            "fixture_error": item.get("error"),
            "source_fuzzy_score": source_score,
            "source_state": raw_result.get("state"),
            "source_dominant_defect": raw_result.get("dominant_defect"),
            "raw_blur": number(raw_metrics.get("blur")),
            "raw_contrast": number(raw_metrics.get("contrast")),
            "raw_snr": number(v2_raw.get("noise")),
            "raw_noise_status": v2_raw_status.get("noise"),
            "v2_blur_severity": blur_sev,
            "v2_contrast_severity": number(v2_metrics.get("contrast")),
            "v2_noise_severity": noise_sev,
            "noise_minus_blur_severity": (
                noise_sev - blur_sev
                if noise_sev is not None and blur_sev is not None
                else None
            ),
            "v2_fuzzy_score": v2_score,
            "v2_state": v2.get("state"),
            "v2_dominant_defect": v2.get("dominant_defect"),
            "source_score_equals_v2_score": (
                source_score == v2_score
                if source_score is not None and v2_score is not None
                else None
            ),
        }
        row["source_candidate_match"] = (
            row["source_dominant_defect"]
            == SOURCE_EXPECTED_LABEL.get(row["reference_class"])
            if row["reference_class"] in SOURCE_EXPECTED_LABEL
            else None
        )
        row["v2_candidate_match"] = (
            row["v2_dominant_defect"]
            == V2_EXPECTED_LABEL.get(row["reference_class"])
            if row["reference_class"] in V2_EXPECTED_LABEL
            else None
        )
        rows.append(row)

    classes = sorted({
        row["reference_class"] for row in rows
        if row["reference_class"] is not None
    })

    by_class = {}
    for cls in classes:
        group = [r for r in rows if r["reference_class"] == cls]
        by_class[cls] = {
            "count": len(group),
            "fixture_status_counts": counts(r["fixture_status"] for r in group),
            "source_fuzzy_score": describe(
                r["source_fuzzy_score"] for r in group
            ),
            "source_state_counts": counts(r["source_state"] for r in group),
            "source_dominant_defect_counts": counts(
                r["source_dominant_defect"] for r in group
            ),
            "raw_blur": describe(r["raw_blur"] for r in group),
            "raw_contrast": describe(r["raw_contrast"] for r in group),
            "raw_linear_snr": describe(r["raw_snr"] for r in group),
            "raw_noise_status_counts": counts(
                r["raw_noise_status"] for r in group
            ),
            "v2_blur_severity": describe(
                r["v2_blur_severity"] for r in group
            ),
            "v2_contrast_severity": describe(
                r["v2_contrast_severity"] for r in group
            ),
            "v2_noise_severity": describe(
                r["v2_noise_severity"] for r in group
            ),
            "v2_noise_minus_blur_severity": describe(
                r["noise_minus_blur_severity"] for r in group
            ),
            "v2_state_counts": counts(r["v2_state"] for r in group),
            "v2_dominant_defect_counts": counts(
                r["v2_dominant_defect"] for r in group
            ),
            "source_candidate_label_matches": sum(
                r["source_candidate_match"] is True for r in group
            ),
            "v2_candidate_label_matches": sum(
                r["v2_candidate_match"] is True for r in group
            ),
        }

    transition_counts = counts(
        f'{r["source_dominant_defect"]} -> {r["v2_dominant_defect"]}'
        for r in rows
    )

    source_eval_rows = [
        {
            "reference_class": r["reference_class"],
            "predicted_label": r["source_dominant_defect"],
        }
        for r in rows
        if r["reference_class"] in SOURCE_EXPECTED_LABEL
    ]
    v2_eval_rows = [
        {
            "reference_class": r["reference_class"],
            "predicted_label": r["v2_dominant_defect"],
        }
        for r in rows
        if r["reference_class"] in V2_EXPECTED_LABEL
    ]

    summary = {
        "status": "EXPLORATORY_SINGLE_SOURCE_NOT_CLINICALLY_VALIDATED",
        "input_report": str(INPUT.relative_to(ROOT)),
        "input_report_status": report.get("status"),
        "total_fixture_count": len(rows),
        "fixture_status_counts": counts(r["fixture_status"] for r in rows),
        "source_fuzzy_score": describe(
            r["source_fuzzy_score"] for r in rows
        ),
        "source_state_counts": counts(r["source_state"] for r in rows),
        "source_dominant_defect_counts": counts(
            r["source_dominant_defect"] for r in rows
        ),
        "v2_fuzzy_score": describe(r["v2_fuzzy_score"] for r in rows),
        "v2_state_counts": counts(r["v2_state"] for r in rows),
        "v2_dominant_defect_counts": counts(
            r["v2_dominant_defect"] for r in rows
        ),
        "source_vs_v2_defect_transitions": transition_counts,
        "source_score_equals_v2_score_count": sum(
            r["source_score_equals_v2_score"] is True for r in rows
        ),
        "by_reference_class": by_class,
        "candidate_label_evaluation": {
            "status": "PROVISIONAL_MAPPING_ONLY_NOT_APPROVED_GROUND_TRUTH",
            "warning": (
                "The noise reference class represents injected noise in this dataset. "
                "Mapping it to sensor_noise is for exploratory counting only and "
                "does not establish physical sensor-noise semantics."
            ),
            "source_m2": [
                one_vs_rest(
                    source_eval_rows, cls, expected_label
                )
                for cls, expected_label in SOURCE_EXPECTED_LABEL.items()
            ],
            "v2": [
                one_vs_rest(
                    v2_eval_rows, cls, expected_label
                )
                for cls, expected_label in V2_EXPECTED_LABEL.items()
            ],
        },
        "limitations": [
            "All results derive from the existing fixture report and do not independently validate the estimator.",
            "The dataset contains one underlying CT source with synthetic degradations.",
            "Source fuzzy quality scoring and v2 dominant-defect selection are separate outputs.",
            "Severity anchors are provisional; class-label mappings require review.",
            "No clinical generalization is claimed.",
        ],
    }

    OUT_JSON.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fields = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Analyzed fixtures: {len(rows)}")
    print(f"Summary JSON: {OUT_JSON}")
    print(f"Per-image CSV: {OUT_CSV}")
    print("Source state counts:", summary["source_state_counts"])
    print("V2 state counts:", summary["v2_state_counts"])
    print("V2 defect counts:", summary["v2_dominant_defect_counts"])
    print("Source-to-v2 defect transitions:", transition_counts)
    print("Candidate mapping: provisional only; see report limitations.")


if __name__ == "__main__":
    main()