"""Generate real M2 v2 fixtures from the project's real source and labeled PNGs.

Run from repository root in the project's venv:
  python scripts/generate_m2_v2_fixtures.py

Writes reports/m2_v2_fixture_generation.json, including successful real outputs and
per-image errors. A failed candidate (e.g. disabled high_contrast trigger) is recorded
as an error, never coerced into a successful payload. Review fixtures/errors before commit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.fuzzy_engine import analyze_image_fuzzy
from cv_fuzzy.v2_contract import ContractV2Error, build_v2_payload

ROOT = Path.cwd()
DATA = ROOT / "data" / "gold_standard"
MANIFEST = DATA / "manifest.json"
SOURCE = ROOT / "cv_fuzzy" / "sample_data" / "CT_small.dcm"
OUT = ROOT / "reports"


def clean_nonfinite(value):
    """Make debug/raw M2 output JSON-safe without implying it is a v2 success payload."""
    if isinstance(value, dict):
        return {k: clean_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_nonfinite(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
    return value


def run_one(label, filename, image):
    raw = analyze_image_fuzzy(image)
    item = {"label": label, "filename": filename, "raw_m2_output": clean_nonfinite(raw)}
    try:
        payload = build_v2_payload(raw, image)
        item["v2_output"] = payload
        item["status"] = "success"
    except (ContractV2Error, ValueError, TypeError) as exc:
        item["status"] = "blocked_error"
        item["error"] = str(exc)
    return item


def main():
    if not MANIFEST.exists() or not SOURCE.exists():
        raise SystemExit("Missing gold-standard manifest or clean source DICOM")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    results.append(run_one("clean_source", str(SOURCE.relative_to(ROOT)), load_dicom(str(SOURCE))))
    for rec in manifest:
        path = DATA / rec["filename"]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            results.append({"label": rec["injected_defect"], "filename": rec["filename"],
                            "status": "blocked_error", "error": "could not read grayscale image"})
            continue
        results.append(run_one(rec["injected_defect"], rec["filename"], image))

    OUT.mkdir(exist_ok=True)
    report = {
        "status": "DRAFT_FIXTURES_NOT_YET_REVIEWED",
        "source_commit_baseline": "8427b0c",
        "generated_from_actual_m2_function": "cv_fuzzy.fuzzy_engine.analyze_image_fuzzy",
        "success_count": sum(x.get("status") == "success" for x in results),
        "blocked_count": sum(x.get("status") != "success" for x in results),
        "note": "Review all blocked errors. Do not send failed candidates to M1 as successful assessments.",
        "fixtures": results,
    }
    output = OUT / "m2_v2_fixture_generation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Generated {len(results)} candidate fixtures: {report['success_count']} successes, "
          f"{report['blocked_count']} blocked.\nReport: {output}")
    if report["blocked_count"]:
        print("WARNING: blocked candidates need review; do not freeze v2 until resolved/documented.")


if __name__ == "__main__":
    main()
