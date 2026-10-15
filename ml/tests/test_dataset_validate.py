"""Regression tests for ml/dataset/validate.py.

Tests verify:
  - Valid rows pass validation.
  - Per-row violations are detected independently (not just on the last row).
  - An invalid early row is caught even when later rows are valid.
  - Malformed JSON fails cleanly.
  - Empty and size-mismatched datasets fail with the row-count check.
  - P10 (cross-metric argmax) is deferred; no argmax assertion is made here.
"""

import json
import os
import sys
import pytest

# Ensure the project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.dataset.validate import validate_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(defect="motion_blur", state="REJECT", blur=80.0, contrast=52.0, noise=20.0,
         output="Blur score high."):
    """Return a minimal valid training row dict."""
    metrics = {
        "blur": blur,
        "contrast": contrast,
        "noise": noise,
        "state": state,
        "dominant_defect": defect,
    }
    return {
        "instruction": "Given DICOM image quality metrics, explain the defect and give a clinical retake instruction.",
        "input": json.dumps(metrics),
        "output": output,
    }


def _write_jsonl(rows, tmpdir):
    path = os.path.join(tmpdir, "test.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Basic passing cases
# ---------------------------------------------------------------------------

def test_single_valid_reject_motion_blur_row_passes(tmp_path):
    rows = [_row(defect="motion_blur", state="REJECT", blur=80.0)]
    path = _write_jsonl(rows, tmp_path)
    # Should not raise
    validate_dataset(str(path), expected_rows=1)


def test_single_valid_accept_none_row_passes(tmp_path):
    rows = [_row(defect="none", state="ACCEPT", blur=20.0, contrast=55.0, noise=12.0)]
    path = _write_jsonl(rows, tmp_path)
    validate_dataset(str(path), expected_rows=1)


def test_multiple_valid_rows_pass(tmp_path):
    rows = [
        _row(defect="motion_blur",  state="REJECT",     blur=80.0,  contrast=52.0, noise=20.0),
        _row(defect="low_contrast", state="REJECT",     blur=15.0,  contrast=20.0, noise=18.0),
        _row(defect="sensor_noise", state="BORDERLINE", blur=25.0,  contrast=54.0, noise=50.0),
        _row(defect="none",         state="ACCEPT",     blur=10.0,  contrast=57.0, noise=12.0),
    ]
    path = _write_jsonl(rows, tmp_path)
    validate_dataset(str(path), expected_rows=4)


# ---------------------------------------------------------------------------
# Per-row violation detection
# ---------------------------------------------------------------------------

def test_early_invalid_row_caught_even_when_later_rows_are_valid(tmp_path):
    """An invalid row at position 1 must be caught even if rows 2 onwards are valid.

    Exercises existing per-row metric boundary validation (Sections 5 & 6 of validate.py),
    not P10 argmax validation. P10 cross-metric argmax validation is deferred.
    Confirms that per-row boundary violations are caught independently of row position.
    """
    rows = [
        # Row 1: REJECT motion_blur but blur is in non-REJECT range (invalid).
        _row(defect="motion_blur", state="REJECT", blur=30.0),   # should fail: 30 not in [65, 100]
        # Row 2: valid
        _row(defect="motion_blur", state="REJECT", blur=80.0),   # should pass
    ]
    path = _write_jsonl(rows, tmp_path)
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=2)


def test_invalid_middle_row_is_caught(tmp_path):
    """A violation in the middle row must be detected regardless of row order.

    Exercises existing per-row metric boundary validation (Sections 5 & 6 of validate.py),
    not P10 argmax validation. P10 cross-metric argmax validation is deferred.
    """
    rows = [
        _row(defect="motion_blur",  state="REJECT",     blur=80.0),  # valid
        _row(defect="low_contrast", state="BORDERLINE", blur=20.0, contrast=90.0),  # invalid: contrast 90 not in [35,49]
        _row(defect="sensor_noise", state="REJECT",     noise=70.0), # valid
    ]
    path = _write_jsonl(rows, tmp_path)
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=3)


def test_last_row_violation_is_caught(tmp_path):
    """A violation only on the last row must still be detected.

    Exercises existing per-row metric boundary validation (Sections 5 & 6 of validate.py),
    not P10 argmax validation. P10 cross-metric argmax validation is deferred.
    """
    rows = [
        _row(defect="none", state="ACCEPT", blur=10.0, contrast=55.0, noise=12.0),  # valid
        _row(defect="motion_blur", state="REJECT", blur=30.0),  # invalid: blur 30 not in [65,100]
    ]
    path = _write_jsonl(rows, tmp_path)
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=2)


# ---------------------------------------------------------------------------
# State <-> defect consistency (legacy-contract checks, pending D7(a) approval)
# ---------------------------------------------------------------------------

def test_accept_row_with_non_none_defect_fails(tmp_path):
    """Legacy-contract check: ACCEPT requires dominant_defect == 'none'. Pending D7(a) approval."""
    rows = [_row(defect="motion_blur", state="ACCEPT", blur=80.0)]
    path = _write_jsonl(rows, tmp_path)
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=1)


def test_reject_row_with_none_defect_fails(tmp_path):
    """Legacy-contract check: REJECT requires dominant_defect != 'none'. Pending D7(a) approval."""
    rows = [_row(defect="none", state="REJECT", blur=80.0)]
    path = _write_jsonl(rows, tmp_path)
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=1)


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------

def test_malformed_json_row_fails_cleanly(tmp_path):
    path = os.path.join(tmp_path, "bad.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json}\n")
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=1)


def test_empty_file_fails_with_row_count_error(tmp_path):
    path = os.path.join(tmp_path, "empty.jsonl")
    open(path, "w").close()
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=1)


def test_row_count_mismatch_fails(tmp_path):
    rows = [_row(defect="none", state="ACCEPT")]
    path = _write_jsonl(rows, tmp_path)
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=5)  # expects 5, gets 1


def test_missing_input_key_fails_cleanly(tmp_path):
    """A row missing the 'input' key must not cause an unbound-variable exception."""
    bad_row = {"instruction": "test", "output": "test"}  # no 'input' key
    path = os.path.join(tmp_path, "missing_key.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(bad_row) + "\n")
    with pytest.raises(SystemExit):
        validate_dataset(str(path), expected_rows=1)
