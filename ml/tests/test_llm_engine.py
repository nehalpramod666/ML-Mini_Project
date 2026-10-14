import math
import pytest
import json
from pathlib import Path

from ml import llm_engine as e


M2_CONTRACT = {
    "metrics": {
        "blur": 108.1619,
        "contrast": 46.9381,
        "noise": 2.0353,
    },
    "fuzzy_score": 32,
    "state": "REJECT",
    "dominant_defect": "motion_blur",
}

FLAT_INPUT = {
    "blur": 85.2,
    "contrast": 40.1,
    "noise": 12.4,
    "state": "REJECT",
    "dominant_defect": "motion_blur",
}

def test_nested_raw_contract_requires_scale_marker():
    raw_contract = {
        "metrics": {
            "blur": 42.0,
            "contrast": 31.2,
            "noise": 18.4,
        },
        "fuzzy_score": 32,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }

    with pytest.raises(e.ContractError, match="metrics_scale"):
        e.to_llm_input(raw_contract)


def test_version_two_calibrated_contract_is_accepted():
    contract = {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {
            "blur": 108.16,
            "contrast": 46.94,
            "noise": 2.04,
        },
        "metrics": {
            "blur": 85.2,
            "contrast": 40.1,
            "noise": 12.4,
        },
        "fuzzy_score": 32,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }

    result = e.to_llm_input(contract)

    assert result == {
        "blur": 85.2,
        "contrast": 40.1,
        "noise": 12.4,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }


def test_version_two_rejects_unapproved_high_noise_alias():
    contract = {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {
            "blur": 100.0,
            "contrast": 30.0,
            "noise": 2.0,
        },
        "metrics": {
            "blur": 85.0,
            "contrast": 30.0,
            "noise": 80.0,
        },
        "fuzzy_score": 32,
        "state": "REJECT",
        "dominant_defect": "high_noise",
    }

    with pytest.raises(e.ContractError, match="not been approved"):
        e.to_llm_input(contract)


def test_version_two_rejects_infinite_calibrated_score():
    contract = {
        "contract_version": 2,
        "metrics_scale": "severity_0_100",
        "metrics_raw": {
            "blur": 100.0,
            "contrast": 30.0,
            "noise": None,
        },
        "metrics": {
            "blur": 85.0,
            "contrast": 30.0,
            "noise": float("inf"),
        },
        "fuzzy_score": 32,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }

    with pytest.raises(e.ContractError, match="finite"):
        e.to_llm_input(contract)

def test_system_prompt_matches_training():
    train_path = (
        Path(__file__).resolve().parents[1]
        / "dataset"
        / "train_chat.jsonl"
    )

    with train_path.open("r", encoding="utf-8") as file:
        row = json.loads(file.readline())

    messages = row.get("messages") or row.get("conversations")
    assert messages is not None, "Training row has no messages"

    training_prompt = next(
        message["content"]
        for message in messages
        if message["role"] == "system"
    )

    assert e.SYSTEM_PROMPT == training_prompt

def test_flat_input_matches_training_shape():
    result = e.to_llm_input(FLAT_INPUT)

    assert list(result) == [
        "blur",
        "contrast",
        "noise",
        "state",
        "dominant_defect",
    ]
    assert result["blur"] == 85.2
    assert "fuzzy_score" not in result


def test_accepts_already_flat_input():
    flat = e.to_llm_input(FLAT_INPUT)

    assert e.to_llm_input(flat) == flat


def test_nested_raw_m2_contract_is_rejected_until_calibrated():
    with pytest.raises(e.ContractError, match="metrics_scale"):
        e.to_llm_input(M2_CONTRACT)


@pytest.mark.parametrize(
    "bad_input",
    [
        {**FLAT_INPUT, "state": "FAIL"},
        {**FLAT_INPUT, "dominant_defect": "high_noise"},
        {**FLAT_INPUT, "dominant_defect": "unknown"},
        {**FLAT_INPUT, "blur": 101.0},
        {**FLAT_INPUT, "contrast": -1.0},
        {**FLAT_INPUT, "noise": float("inf")},
        {**FLAT_INPUT, "blur": float("nan")},
        {**FLAT_INPUT, "noise": -float("inf")},
    ],
)
def test_invalid_flat_input_raises_contract_error(bad_input):
    with pytest.raises(e.ContractError):
        e.to_llm_input(bad_input)


def test_nonfinite_nested_metric_is_rejected():
    bad_contract = {
        "contract_version": 2,
        "metrics_scale": e.SUPPORTED_SCALE,
        "metrics_raw": {
            "blur": 108.16,
            "contrast": 46.94,
            "noise": float("inf"),
        },
        "metrics": {
            "blur": 85.2,
            "contrast": 40.1,
            "noise": 12.4,
        },
        "fuzzy_score": 32,
        "state": "REJECT",
        "dominant_defect": "motion_blur",
    }

    with pytest.raises(e.ContractError, match="finite"):
        e.to_llm_input(bad_contract)


@pytest.mark.parametrize(
    "bad_contract",
    [
        {**M2_CONTRACT, "state": "FAIL"},
        {**M2_CONTRACT, "dominant_defect": "sensor_noise"},
        {
            **M2_CONTRACT,
            "fuzzy_score": 101,
        },
        {
            **M2_CONTRACT,
            "metrics": {"blur": 1.0, "contrast": 2.0},
        },
    ],
)
def test_invalid_nested_contract_raises(bad_contract):
    with pytest.raises(e.ContractError):
        e.to_llm_input(bad_contract)


def test_import_does_not_load_model():
    assert e._llm is None


def test_messages_use_flat_json_without_nonfinite_values():
    messages = e._messages(FLAT_INPUT)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Parsing this proves that the generated content is valid JSON.
    import json

    payload = json.loads(messages[1]["content"])

    assert list(payload) == [
        "blur",
        "contrast",
        "noise",
        "state",
        "dominant_defect",
    ]
    assert all(
        math.isfinite(payload[field])
        for field in ("blur", "contrast", "noise")
    )