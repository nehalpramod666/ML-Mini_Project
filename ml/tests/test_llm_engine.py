import copy
import json
import math
from pathlib import Path
import pytest

from cv_fuzzy.dicom_loader import load_dicom
from cv_fuzzy.fuzzy_engine import analyze_image_fuzzy
from cv_fuzzy.v2_contract import build_v2_payload
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


def _real_dicom_v2_payload():
    dcm_path = Path(__file__).resolve().parents[2] / "cv_fuzzy" / "sample_data" / "CT_small.dcm"
    img = load_dicom(str(dcm_path))
    m2_res = analyze_image_fuzzy(img)
    return build_v2_payload(m2_res, img)


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
    contract = _real_dicom_v2_payload()
    result = e.to_llm_input(contract)

    assert result == {
        "blur": contract["metrics"]["blur"],
        "contrast": contract["metrics"]["contrast"],
        "noise": contract["metrics"]["noise"],
        "state": "REJECT",
        "dominant_defect": "sensor_noise",
    }
    assert result["blur"] == 0
    assert result["contrast"] == 0
    assert result["noise"] == 80


def test_version_two_canonical_low_snr_maps_to_sensor_noise():
    """Canonical v2 label 'low_snr' is explicitly adapted to model vocabulary 'sensor_noise'."""
    contract = _real_dicom_v2_payload()
    assert contract["dominant_defect"] == "low_snr"

    result = e.to_llm_input(contract)
    assert result["dominant_defect"] == "sensor_noise"


def test_version_two_rejects_nested_sensor_noise_as_canonical():
    """Nested v2 payload must use canonical 'low_snr', not internal training label 'sensor_noise'."""
    contract = _real_dicom_v2_payload()
    contract["dominant_defect"] = "sensor_noise"

    with pytest.raises(e.ContractError, match="canonical label 'low_snr'"):
        e.to_llm_input(contract)


def test_version_two_rejects_unapproved_high_noise_alias():
    contract = _real_dicom_v2_payload()
    contract["dominant_defect"] = "high_noise"

    with pytest.raises(e.ContractError, match="rejected and cannot be mapped"):
        e.to_llm_input(contract)


def test_version_two_rejects_fractional_severities():
    contract = _real_dicom_v2_payload()
    contract["metrics"]["blur"] = 12.5  # fractional float rejected

    with pytest.raises(e.ContractError, match="must be an integer"):
        e.to_llm_input(contract)


def test_version_two_rejects_boolean_severities():
    contract = _real_dicom_v2_payload()
    contract["metrics"]["blur"] = True  # boolean rejected

    with pytest.raises(e.ContractError, match="must be an integer"):
        e.to_llm_input(contract)


def test_version_two_rejects_infinite_calibrated_score():
    contract = _real_dicom_v2_payload()
    contract["metrics"]["noise"] = float("inf")

    with pytest.raises(e.ContractError, match="must be an integer"):
        e.to_llm_input(contract)


def test_malformed_nested_v2_payloads_rejected():
    base_payload = _real_dicom_v2_payload()

    # 1. Missing required top-level field (metrics_raw_status)
    bad_missing = copy.deepcopy(base_payload)
    del bad_missing["metrics_raw_status"]
    with pytest.raises(e.ContractError, match="missing or unexpected"):
        e.to_llm_input(bad_missing)

    # 2. Unexpected extra top-level field
    bad_extra = copy.deepcopy(base_payload)
    bad_extra["extra_field"] = "unexpected"
    with pytest.raises(e.ContractError, match="missing or unexpected"):
        e.to_llm_input(bad_extra)

    # 3. State mismatch with fuzzy_score
    bad_state = copy.deepcopy(base_payload)
    bad_state["state"] = "ACCEPT"  # Score is 31, which is REJECT
    with pytest.raises(e.ContractError, match="state disagrees"):
        e.to_llm_input(bad_state)

    # 4. Inconsistent calibrated metric value (violates formula)
    bad_metric = copy.deepcopy(base_payload)
    bad_metric["metrics"]["noise"] = 50  # expected is 80
    with pytest.raises(e.ContractError, match="disagrees with documented provisional conversion"):
        e.to_llm_input(bad_metric)

    # 5. Dominant defect inconsistent with severity candidates
    bad_defect = copy.deepcopy(base_payload)
    bad_defect["dominant_defect"] = "motion_blur"  # noise=80 > blur=0
    with pytest.raises(e.ContractError, match="dominant_defect disagrees"):
        e.to_llm_input(bad_defect)

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
    bad_contract = _real_dicom_v2_payload()
    bad_contract["metrics_raw"]["noise"] = float("inf")

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


def test_messages_neutral_contextual_instruction_for_sensor_noise():
    """Verify neutral SNR measurement instructions are added for sensor_noise without breaking base prompt parity."""
    # Base system prompt remains untouched
    assert e.SYSTEM_PROMPT == (
        "You are a clinical imaging quality assistant. Given DICOM quality metrics "
        "(blur, contrast, noise scores, overall state, and dominant defect), explain "
        "the defect and give a concise, actionable retake instruction."
    )

    # When dominant_defect is sensor_noise, the neutral contextual instruction is appended
    sensor_noise_input = {**FLAT_INPUT, "dominant_defect": "sensor_noise"}
    messages = e._messages(sensor_noise_input)
    assert messages[0]["role"] == "system"
    assert e.SNR_CONTEXTUAL_INSTRUCTION in messages[0]["content"]
    assert "low signal-to-noise ratio" in messages[0]["content"]
    assert "not confirmed physical sensor hardware malfunction" in messages[0]["content"]

    # When dominant_defect is NOT sensor_noise, the base prompt is used directly
    motion_blur_messages = e._messages(FLAT_INPUT)
    assert motion_blur_messages[0]["content"] == e.SYSTEM_PROMPT
