from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv_fuzzy.metrics import analyze_image
from cv_fuzzy.fuzzy_engine import analyze_image_fuzzy
from cv_fuzzy.v2_contract import build_v2_payload


MANIFEST_PATH = ROOT / "data" / "gold_standard" / "manifest.json"
IMAGE_DIR = ROOT / "data" / "gold_standard"
OUTPUT_DIR = ROOT / "reports" / "stage1_baseline_20261027"
RULES_PATH = OUTPUT_DIR / "scoring_rules.md"

EXPECTED_LABEL = {
    "blur": "motion_blur",
    "contrast": "low_contrast",
    "noise": "low_snr",
}

EXPECTED_LABELS = ["motion_blur", "low_contrast", "low_snr"]

OUTPUT_FILES = [
    OUTPUT_DIR / "per_image.csv",
    OUTPUT_DIR / "summary.json",
    OUTPUT_DIR / "confusion_matrix.csv",
]

CSV_FIELDS = [
    "filename",
    "manifest_index",
    "injected_defect",
    "expected_v2_label",
    "predicted_v2_label",
    "is_isolated",
    "result_status",
    "manifest_source",
    "parameters_json",
    "manifest_seed",
    "metrics_function_blur",
    "metrics_function_contrast",
    "metrics_function_noise",
    "raw_blur",
    "raw_contrast",
    "raw_noise",
    "metrics_consistent_with_fuzzy",
    "fuzzy_score",
    "fuzzy_state",
    "legacy_dominant_defect",
    "v2_blur_severity",
    "v2_contrast_severity",
    "v2_noise_severity",
    "v2_fuzzy_score",
    "v2_state",
    "v2_dominant_defect",
    "error_stage",
    "error_type",
    "error_code",
    "error_message",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    """Make values safe for standards-compliant JSON serialization."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def number_or_blank(value):
    if value is None:
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def same_number(left, right) -> bool:
    try:
        left = float(left)
        right = float(right)
    except (TypeError, ValueError):
        return False

    if math.isnan(left) or math.isnan(right):
        return False

    if math.isinf(left) or math.isinf(right):
        return left == right

    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)


def initial_row(item: dict, index: int) -> dict:
    defect = str(item.get("injected_defect", "")).strip().lower()

    return {
        "filename": str(item.get("filename", "")),
        "manifest_index": index,
        "injected_defect": defect,
        "expected_v2_label": EXPECTED_LABEL[defect],
        "predicted_v2_label": "ERROR",
        "is_isolated": False,
        "result_status": "not_processed",
        "manifest_source": str(item.get("source", "")),
        "parameters_json": json.dumps(
            item.get("parameters", {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "manifest_seed": item.get("seed", ""),
        "metrics_function_blur": "",
        "metrics_function_contrast": "",
        "metrics_function_noise": "",
        "raw_blur": "",
        "raw_contrast": "",
        "raw_noise": "",
        "metrics_consistent_with_fuzzy": "",
        "fuzzy_score": "",
        "fuzzy_state": "",
        "legacy_dominant_defect": "",
        "v2_blur_severity": "",
        "v2_contrast_severity": "",
        "v2_noise_severity": "",
        "v2_fuzzy_score": "",
        "v2_state": "",
        "v2_dominant_defect": "",
        "error_stage": "",
        "error_type": "",
        "error_code": "",
        "error_message": "",
    }


def record_error(row: dict, stage: str, exc: Exception) -> None:
    row["result_status"] = f"{stage}_error"
    row["error_stage"] = stage
    row["error_type"] = type(exc).__name__
    row["error_code"] = str(getattr(exc, "code", "") or "")
    row["error_message"] = str(exc)
    row["predicted_v2_label"] = "ERROR"
    row["is_isolated"] = False


def count_values(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(
        Counter(
            str(row[field])
            for row in rows
            if row.get(field) not in ("", None)
        ).items()
    ))


def main() -> int:
    # Refuse to run unless the scoring rules were saved first.
    if not RULES_PATH.is_file():
        print(f"ERROR: scoring rules missing: {RULES_PATH}", file=sys.stderr)
        return 2

    # Refuse to overwrite any previous baseline outputs.
    existing = [str(path) for path in OUTPUT_FILES if path.exists()]
    if existing:
        print("ERROR: baseline output files already exist.", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        print("Inspect the existing files; this script will not overwrite them.")
        return 2

    if not MANIFEST_PATH.is_file():
        print(f"ERROR: manifest missing: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"ERROR: could not read manifest: {exc}", file=sys.stderr)
        return 2

    if not isinstance(manifest, list) or len(manifest) != 100:
        print(
            "ERROR: expected the verified 100-record manifest list.",
            file=sys.stderr,
        )
        return 2

    filenames = [item.get("filename") for item in manifest if isinstance(item, dict)]

    if len(filenames) != len(manifest) or any(not name for name in filenames):
        print("ERROR: manifest contains invalid records or filenames.", file=sys.stderr)
        return 2

    if len(set(filenames)) != len(filenames):
        print("ERROR: duplicate manifest filenames detected.", file=sys.stderr)
        return 2

    unknown_defects = sorted({
        str(item.get("injected_defect", "")).strip().lower()
        for item in manifest
        if str(item.get("injected_defect", "")).strip().lower()
        not in EXPECTED_LABEL
    })

    if unknown_defects:
        print(
            f"ERROR: frozen mapping does not cover: {unknown_defects}",
            file=sys.stderr,
        )
        return 2

    rules_hash = sha256_file(RULES_PATH)
    manifest_hash = sha256_file(MANIFEST_PATH)

    print("Stage 1 exploratory baseline")
    print(f"Manifest records: {len(manifest)}")
    print(f"Frozen scoring rules SHA-256: {rules_hash}")
    print("Production rules and thresholds remain unchanged.")
    print("High-contrast trigger remains disabled.")
    print("The Day 18 locked evaluation will not be accessed.")

    results = []

    for index, item in enumerate(manifest, start=1):
        row = initial_row(item, index)
        image_path = IMAGE_DIR / str(item["filename"])

        print(f"[{index:03d}/{len(manifest)}] {item['filename']}")

        # Stage 1: load this PNG.
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(
                    f"Could not load grayscale PNG: {image_path}"
                )
        except Exception as exc:
            record_error(row, "image_load", exc)
            results.append(row)
            continue

        # Stage 2: explicitly run the existing metric analysis function.
        try:
            metric_result = analyze_image(image)
            direct_metrics = metric_result.get("metrics")

            if not isinstance(direct_metrics, dict):
                raise ValueError("analyze_image returned no metrics dictionary")

            for name in ("blur", "contrast", "noise"):
                row[f"metrics_function_{name}"] = number_or_blank(
                    direct_metrics.get(name)
                )
        except Exception as exc:
            record_error(row, "metrics", exc)
            results.append(row)
            continue

        # Stage 3: run the existing fuzzy inference engine.
        try:
            fuzzy_result = analyze_image_fuzzy(image)

            if not isinstance(fuzzy_result, dict):
                raise ValueError(
                    "analyze_image_fuzzy returned a non-dictionary result"
                )

            fuzzy_metrics = fuzzy_result.get("metrics")
            if not isinstance(fuzzy_metrics, dict):
                raise ValueError(
                    "Fuzzy result has no metrics dictionary"
                )

            for name in ("blur", "contrast", "noise"):
                row[f"raw_{name}"] = number_or_blank(
                    fuzzy_metrics.get(name)
                )

            row["metrics_consistent_with_fuzzy"] = all(
                same_number(direct_metrics.get(name), fuzzy_metrics.get(name))
                for name in ("blur", "contrast", "noise")
            )

            row["fuzzy_score"] = fuzzy_result.get("fuzzy_score", "")
            row["fuzzy_state"] = fuzzy_result.get("state", "")
            row["legacy_dominant_defect"] = (
                fuzzy_result.get("dominant_defect", "")
            )

        except Exception as exc:
            record_error(row, "fuzzy_inference", exc)
            results.append(row)
            continue

        # Stage 4: build the V2 payload using the existing default configuration.
        # Intentionally do not enable the high-contrast trigger.
        try:
            payload = build_v2_payload(fuzzy_result, image)

            if not isinstance(payload, dict):
                raise ValueError("V2 payload builder returned a non-dictionary")

            v2_metrics = payload.get("metrics")
            if not isinstance(v2_metrics, dict):
                raise ValueError("V2 payload contains no metrics dictionary")

            predicted = payload.get("dominant_defect")
            if not isinstance(predicted, str) or not predicted:
                raise ValueError("V2 payload contains no dominant-defect label")

            row["v2_blur_severity"] = v2_metrics.get("blur", "")
            row["v2_contrast_severity"] = v2_metrics.get("contrast", "")
            row["v2_noise_severity"] = v2_metrics.get("noise", "")
            row["v2_fuzzy_score"] = payload.get("fuzzy_score", "")
            row["v2_state"] = payload.get("state", "")
            row["v2_dominant_defect"] = predicted
            row["predicted_v2_label"] = predicted
            row["is_isolated"] = (
                predicted == row["expected_v2_label"]
            )
            row["result_status"] = "success"

        except Exception as exc:
            record_error(row, "v2_payload", exc)

        results.append(row)

    # Verify that the frozen scoring rules did not change during the run.
    if sha256_file(RULES_PATH) != rules_hash:
        print(
            "ERROR: scoring-rules file changed during evaluation. "
            "Do not treat these results as a valid baseline.",
            file=sys.stderr,
        )
        return 3

    if len(results) != 100:
        print(
            f"ERROR: expected 100 result records, got {len(results)}.",
            file=sys.stderr,
        )
        return 3

    # Metrics and V2 payload errors remain in the denominator.
    total = len(results)
    isolated_total = sum(row["is_isolated"] is True for row in results)
    overall_rate = 100.0 * isolated_total / total

    by_defect = {}
    for defect, expected_label in EXPECTED_LABEL.items():
        subset = [row for row in results if row["injected_defect"] == defect]
        isolated = sum(row["is_isolated"] is True for row in subset)
        error_count = sum(bool(row["error_stage"]) for row in subset)

        by_defect[defect] = {
            "expected_v2_label": expected_label,
            "total": len(subset),
            "isolated": isolated,
            "not_isolated": len(subset) - isolated,
            "errors": error_count,
            "isolation_rate_percent": round(
                100.0 * isolated / len(subset), 4
            ) if subset else 0.0,
        }

    observed_labels = {
        str(row["predicted_v2_label"]) for row in results
    }
    prediction_labels = sorted(
        observed_labels.union(EXPECTED_LABELS).union({"ERROR"})
    )

    confusion = {}
    for expected_label in EXPECTED_LABELS:
        confusion[expected_label] = {
            predicted_label: sum(
                row["expected_v2_label"] == expected_label
                and row["predicted_v2_label"] == predicted_label
                for row in results
            )
            for predicted_label in prediction_labels
        }

    errors = [
        {
            "filename": row["filename"],
            "injected_defect": row["injected_defect"],
            "expected_v2_label": row["expected_v2_label"],
            "error_stage": row["error_stage"],
            "error_type": row["error_type"],
            "error_code": row["error_code"],
            "error_message": row["error_message"],
        }
        for row in results
        if row["error_stage"]
    ]

    fuzzy_states = count_values(results, "fuzzy_state")
    v2_states = count_values(results, "v2_state")
    legacy_labels = count_values(results, "legacy_dominant_defect")
    v2_labels = count_values(results, "v2_dominant_defect")

    error_stages = count_values(
        [row for row in results if row["error_stage"]],
        "error_stage",
    )
    error_codes = dict(sorted(Counter(
        row["error_code"] or "UNSPECIFIED"
        for row in results
        if row["error_stage"]
    ).items()))

    fuzzy_success_count = sum(bool(row["fuzzy_state"]) for row in results)
    payload_success_count = sum(
        row["result_status"] == "success" for row in results
    )

    explicit_controls = [
        row for row in manifest
        if any(
            marker in str(row.get("injected_defect", "")).lower()
            for marker in ("clean", "original", "control", "none")
        )
    ]

    summary = {
        "status": "EXPLORATORY_BASELINE",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
            "manifest_sha256": manifest_hash,
            "record_count": total,
            "defect_counts": dict(sorted(Counter(
                row["injected_defect"] for row in results
            ).items())),
            "unique_manifest_sources": sorted({
                str(item.get("source", "")) for item in manifest
            }),
            "explicit_clean_control_count": len(explicit_controls),
        },
        "scoring_rules": {
            "path": str(RULES_PATH.relative_to(ROOT)),
            "sha256": rules_hash,
            "frozen_mapping": EXPECTED_LABEL,
            "criterion": (
                "Exact match between predicted V2 dominant-defect label "
                "and the frozen expected label"
            ),
            "prd_target_percent": 100.0,
            "errors_count_as_not_isolated": True,
        },
        "overall_isolation": {
            "isolated": isolated_total,
            "total": total,
            "not_isolated": total - isolated_total,
            "isolation_rate_percent": round(overall_rate, 4),
        },
        "isolation_by_defect": by_defect,
        "processing": {
            "metric_function_success_count": sum(
                bool(row["metrics_function_blur"] != "")
                for row in results
            ),
            "fuzzy_success_count": fuzzy_success_count,
            "v2_payload_success_count": payload_success_count,
            "error_count": len(errors),
            "errors_by_stage": error_stages,
            "errors_by_code": error_codes,
            "direct_metrics_vs_fuzzy_metrics_mismatch_count": sum(
                row["metrics_consistent_with_fuzzy"] is False
                for row in results
            ),
        },
        "state_distribution": {
            "legacy_fuzzy_state": fuzzy_states,
            "v2_state": v2_states,
        },
        "dominant_defect_distribution": {
            "legacy_fuzzy_label": legacy_labels,
            "v2_label_successes_only": v2_labels,
        },
        "confusion_matrix": {
            "rows": "expected V2 defect label",
            "columns": "predicted V2 defect label; ERROR denotes a failed result",
            "column_labels": prediction_labels,
            "counts": confusion,
        },
        "errors": errors,
        "limitations": [
            "Exploratory synthetic-fixture baseline; not clinical validation.",
            "All fixtures come from the same recorded source image.",
            "The manifest has no explicitly marked clean/original controls.",
            "Isolation is measured against the predeclared mapping, not expert labels.",
            "The production high-contrast trigger remains disabled.",
            "This script does not change or rerun the locked Day 18 evaluation.",
        ],
    }

    # Write only new output files. Existing output files were checked before run.
    with OUTPUT_FILES[0].open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    with OUTPUT_FILES[2].open("x", newline="", encoding="utf-8-sig") as file:
        fields = ["expected_v2_label"] + prediction_labels
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for expected_label in EXPECTED_LABELS:
            writer.writerow({
                "expected_v2_label": expected_label,
                **confusion[expected_label],
            })

    with OUTPUT_FILES[1].open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")

    print("\n=== STAGE 1 EXPLORATORY BASELINE SUMMARY ===")
    print(f"Status: {summary['status']}")
    print(f"Images processed: {total}")
    print(f"Fuzzy inference successful: {fuzzy_success_count}")
    print(f"V2 payload successful: {payload_success_count}")
    print(f"Errors: {len(errors)}")
    print(
        f"Overall isolation: {isolated_total}/{total} "
        f"({overall_rate:.4f}%)"
    )
    print(f"PRD target: {summary['scoring_rules']['prd_target_percent']:.1f}%")

    print("\nIsolation by defect:")
    for defect, result in by_defect.items():
        print(
            f"  {defect}: {result['isolated']}/{result['total']} "
            f"({result['isolation_rate_percent']:.4f}%), "
            f"errors={result['errors']}"
        )

    print("\nLegacy fuzzy states:")
    print(json.dumps(fuzzy_states, indent=2))

    print("\nV2 states:")
    print(json.dumps(v2_states, indent=2))

    print("\nConfusion matrix (JSON counts):")
    print(json.dumps(confusion, indent=2))

    print("\nErrors:")
    if errors:
        for error in errors:
            print(json.dumps(error, ensure_ascii=False))
    else:
        print("None")

    print("\nFiles created:")
    for path in OUTPUT_FILES:
        print(f"  {path.relative_to(ROOT)}")

    print("\nProduction rules, thresholds, and the locked Day 18 evaluation were not modified.")
    print("This is an exploratory baseline, not a clinical validation result.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
