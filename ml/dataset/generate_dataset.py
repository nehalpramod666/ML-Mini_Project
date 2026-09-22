"""
Synthetic dataset generator for clinical retake-instruction fine-tuning.
Schema: see schema.md
"""
import json
import random
from pathlib import Path

DEFECT_TYPES = ["motion_blur", "low_contrast", "high_contrast", "sensor_noise", "none"]
STATES = ["ACCEPT", "BORDERLINE", "REJECT"]

def sample_metrics() -> dict:
    raise NotImplementedError

def build_instruction_text(metrics: dict) -> str:
    raise NotImplementedError

def generate_row() -> dict:
    metrics = sample_metrics()
    output = build_instruction_text(metrics)
    return {
        "instruction": "Given DICOM image quality metrics, explain the defect and give a clinical retake instruction.",
        "input": json.dumps(metrics),
        "output": output,
    }

def main(n_rows: int, out_path: str):
    rows = [generate_row() for _ in range(n_rows)]
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main(n_rows=10, out_path="ml/dataset/smoke_test.jsonl")
    