"""
Synthetic dataset generator for clinical retake-instruction fine-tuning.

Schema: see schema.md

NOTE:
The low/high contrast mapping below is a synthetic proxy for dataset
generation. The final interpretation must be confirmed against M2's
real CV/fuzzy engine.
"""

import json
import random


DEFECT_TYPES = [
    "motion_blur",
    "low_contrast",
    "high_contrast",
    "sensor_noise",
    "none",
]

STATES = [
    "ACCEPT",
    "BORDERLINE",
    "REJECT",
]


def sample_metrics() -> dict:
    """
    Generate one randomized synthetic metrics dictionary.

    The state, dominant defect, and dominant metric are kept internally
    consistent.

    Synthetic contrast-direction assumption:
        low_contrast  -> lower contrast values
        high_contrast -> higher contrast values

    These thresholds are synthetic dataset-generation rules and must be
    confirmed against M2's real CV/fuzzy engine.
    """

    state = random.choice(STATES)

    # ACCEPT rows have no dominant defect.
    defect = (
        "none"
        if state == "ACCEPT"
        else random.choice(DEFECT_TYPES[:-1])
    )

    # Use clearly separated non-defect ranges.
    if state == "ACCEPT":
        metrics = {
            "blur": round(random.uniform(0, 30), 1),
            "contrast": round(random.uniform(50, 59), 1),
            "noise": round(random.uniform(0, 30), 1),
        }

    elif state == "BORDERLINE":
        metrics = {
            "blur": round(random.uniform(20, 39), 1),
            "contrast": round(random.uniform(50, 59), 1),
            "noise": round(random.uniform(20, 39), 1),
        }

    else:  # REJECT
        metrics = {
            "blur": round(random.uniform(10, 39), 1),
            "contrast": round(random.uniform(50, 59), 1),
            "noise": round(random.uniform(10, 39), 1),
        }

    # Put the dominant defect's metric into its intended range.
    if defect == "motion_blur":
        if state == "REJECT":
            metrics["blur"] = round(random.uniform(65, 100), 1)
        else:
            metrics["blur"] = round(random.uniform(40, 64), 1)

    elif defect == "low_contrast":
        if state == "REJECT":
            metrics["contrast"] = round(random.uniform(0, 34), 1)
        else:
            metrics["contrast"] = round(random.uniform(35, 49), 1)

    elif defect == "high_contrast":
        if state == "REJECT":
            metrics["contrast"] = round(random.uniform(65, 100), 1)
        else:
            metrics["contrast"] = round(random.uniform(60, 64), 1)

    elif defect == "sensor_noise":
        if state == "REJECT":
            metrics["noise"] = round(random.uniform(65, 100), 1)
        else:
            metrics["noise"] = round(random.uniform(40, 64), 1)

    return {
        "blur": metrics["blur"],
        "contrast": metrics["contrast"],
        "noise": metrics["noise"],
        "state": state,
        "dominant_defect": defect,
    }


TEMPLATES = {
    "motion_blur": [
        (
            "Motion blur is the dominant image-quality defect "
            "(blur score {blur:.0f}/100). "
            "Keep the patient still during exposure and repeat the acquisition."
        ),
        (
            "A blur score of {blur:.0f}/100 indicates {severity} motion artifact. "
            "Stabilize the patient and minimize movement before repeating the image."
        ),
        (
            "The image shows {severity} motion-related blur "
            "with a score of {blur:.0f}/100. "
            "Reposition as needed, ensure patient stability, and retake the image."
        ),
        (
            "Motion artifact is reducing image sharpness "
            "(blur score {blur:.0f}/100). "
            "Instruct the patient to remain still and repeat the acquisition."
        ),
    ],

    "low_contrast": [
        (
            "Low contrast is the dominant quality defect "
            "(contrast score {contrast:.0f}/100). "
            "Review the exposure technique and repeat the acquisition."
        ),
        (
            "The image shows {severity} low contrast "
            "at a score of {contrast:.0f}/100. "
            "Adjust the acquisition technique according to the imaging protocol."
        ),
        (
            "The contrast score is {contrast:.0f}/100, indicating reduced contrast. "
            "Review the exposure settings and repeat the image."
        ),
    ],

    "high_contrast": [
        (
            "High contrast is the dominant quality defect "
            "(contrast score {contrast:.0f}/100). "
            "Review the exposure technique and repeat the acquisition."
        ),
        (
            "The image shows {severity} high contrast "
            "at a score of {contrast:.0f}/100. "
            "Review the acquisition settings before retaking the image."
        ),
        (
            "The contrast score is {contrast:.0f}/100, indicating elevated contrast. "
            "Review the exposure settings and repeat the acquisition."
        ),
    ],

    "sensor_noise": [
        (
            "Sensor noise is affecting image quality "
            "(noise score {noise:.0f}/100). "
            "Check the detector system and repeat the acquisition if image detail is insufficient."
        ),
        (
            "The noise score of {noise:.0f}/100 indicates {severity} signal degradation. "
            "Inspect the detector for artifacts before retaking the image."
        ),
        (
            "Image noise is contributing to reduced diagnostic clarity "
            "(noise score {noise:.0f}/100). "
            "Verify detector performance and repeat the acquisition if needed."
        ),
    ],

    "none": [
        (
            "Image quality is acceptable across all measured parameters "
            "(blur {blur:.0f}, contrast {contrast:.0f}, noise {noise:.0f}). "
            "No retake is required."
        ),
        (
            "All reported quality metrics are within the acceptable range. "
            "No retake is needed."
        ),
        (
            "No significant image-quality defect is identified "
            "(blur {blur:.0f}, contrast {contrast:.0f}, noise {noise:.0f}). "
            "The image can proceed without retake."
        ),
        (
            "The reported metrics indicate acceptable image quality. "
            "No corrective acquisition is required."
        ),
    ],
}


def _severity_word(value: float) -> str:
    if value >= 80:
        return "severe"

    if value >= 65:
        return "moderate"

    if value >= 40:
        return "mild"

    return "minimal"


def build_instruction_text(metrics: dict) -> str:
    """Build an instruction from metrics using varied templates."""

    defect = metrics["dominant_defect"]

    template = random.choice(
        TEMPLATES[defect]
    )

    defect_value = {
        "motion_blur": metrics["blur"],
        "low_contrast": 100 - metrics["contrast"],
        "high_contrast": metrics["contrast"],
        "sensor_noise": metrics["noise"],
        "none": 0,
    }[defect]

    return template.format(
        severity=_severity_word(defect_value),
        blur=metrics["blur"],
        contrast=metrics["contrast"],
        noise=metrics["noise"],
    )


def generate_row() -> dict:
    metrics = sample_metrics()
    output = build_instruction_text(metrics)

    return {
        "instruction": (
            "Given DICOM image quality metrics, explain the defect "
            "and give a clinical retake instruction."
        ),
        "input": json.dumps(metrics),
        "output": output,
    }


def main(n_rows: int, out_path: str):
    rows = [
        generate_row()
        for _ in range(n_rows)
    ]

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(row) + "\n"
            )

    print(
        f"Wrote {n_rows} rows to {out_path}"
    )


if __name__ == "__main__":
    main(
        n_rows=500,
        out_path="ml/dataset/train_500.jsonl",
    )