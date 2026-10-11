"""Fuzzy image-quality scoring for the M2 computer-vision pipeline."""


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from cv_fuzzy.metrics import analyze_image


# Initial engineering ranges for the current synthetic calibration dataset.
# These are not clinically validated thresholds.
BLUR_MAX = 25000.0
CONTRAST_MAX = 80.0
SNR_MAX = 10.0


def _build_control_system():
    """Build the fuzzy variables and inference rules."""
    blur_universe = np.unique(
        np.concatenate(
            (
                np.linspace(0, 200, 401),
                np.linspace(200, BLUR_MAX, 500),
            )
        )
    )

    blur = ctrl.Antecedent(blur_universe, "blur")
    contrast = ctrl.Antecedent(
        np.linspace(0, CONTRAST_MAX, 401), "contrast"
    )
    snr = ctrl.Antecedent(np.linspace(0, SNR_MAX, 201), "snr")
    quality = ctrl.Consequent(np.linspace(0, 100, 101), "quality")

    blur["low"] = fuzz.trapmf(
        blur.universe, [0, 0, 5, 50]
    )
    blur["medium"] = fuzz.trimf(
        blur.universe, [20, 60, 110]
    )
    blur["high"] = fuzz.trapmf(
        blur.universe, [80, 110, BLUR_MAX, BLUR_MAX]
    )

    contrast["low"] = fuzz.trapmf(
        contrast.universe, [0, 0, 20, 38]
    )
    contrast["medium"] = fuzz.trimf(
        contrast.universe, [15, 35, 55]
    )
    contrast["high"] = fuzz.trapmf(
        contrast.universe, [40, 60, CONTRAST_MAX, CONTRAST_MAX]
    )

    # compute_noise() returns an SNR estimate: lower is noisier.
    snr["low"] = fuzz.trapmf(snr.universe, [0, 0, 1.5, 2.5])
    snr["medium"] = fuzz.trimf(snr.universe, [1.5, 3.5, 6])
    snr["high"] = fuzz.trapmf(
        snr.universe, [4, 7, SNR_MAX, SNR_MAX]
    )

    quality["poor"] = fuzz.trapmf(
        quality.universe, [0, 0, 20, 45]
    )
    quality["fair"] = fuzz.trimf(quality.universe, [25, 50, 75])
    quality["good"] = fuzz.trapmf(
        quality.universe, [55, 80, 100, 100]
    )

    rules = [
        ctrl.Rule(blur["low"], quality["poor"]),
        ctrl.Rule(contrast["low"], quality["poor"]),
        ctrl.Rule(snr["low"], quality["poor"]),
        ctrl.Rule(
            blur["medium"] & contrast["medium"] & snr["medium"],
            quality["fair"],
        ),
        ctrl.Rule(
            blur["high"] & contrast["medium"] & snr["high"],
            quality["good"],
        ),
        ctrl.Rule(
            blur["high"] & contrast["high"] & snr["high"],
            quality["good"],
        ),
        ctrl.Rule(
            blur["medium"] & contrast["high"] & snr["high"],
            quality["good"],
        ),
    ]

    return ctrl.ControlSystem(rules), blur, contrast, snr


_CONTROL_SYSTEM, _BLUR, _CONTRAST, _SNR = _build_control_system()


def analyze_image_fuzzy(img: np.ndarray) -> dict:
    """Return image metrics, a fuzzy quality score, state, and main defect."""
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale array, got shape {img.shape}"
        )
    if img.size == 0:
        raise ValueError("img must not be empty")

    analysis = analyze_image(img)
    metrics = analysis["metrics"]

    blur_value = float(metrics["blur"])
    contrast_value = float(metrics["contrast"])
    snr_value = float(metrics["noise"])

    # Keep fuzzy inputs within their defined universes.
    blur_input = float(np.clip(blur_value, 0, BLUR_MAX))
    contrast_input = float(np.clip(contrast_value, 0, CONTRAST_MAX))
    snr_input = SNR_MAX if np.isposinf(snr_value) else float(
        np.clip(snr_value, 0, SNR_MAX)
    )
    if not np.isfinite(snr_input):
        snr_input = 0.0

    simulation = ctrl.ControlSystemSimulation(_CONTROL_SYSTEM)
    simulation.input["blur"] = blur_input
    simulation.input["contrast"] = contrast_input
    simulation.input["snr"] = snr_input
    simulation.compute()

    score = int(round(float(np.clip(simulation.output["quality"], 0, 100))))

    if score >= 75:
        state = "ACCEPT"
    elif score >= 45:
        state = "BORDERLINE"
    else:
        state = "REJECT"

    # Select the strongest measured defect membership.
    defect_memberships = {
        "motion_blur": float(
            fuzz.interp_membership(
                _BLUR.universe,
                _BLUR["low"].mf,
                blur_input,
            )
        ),
        "low_contrast": float(
            fuzz.interp_membership(
                np.linspace(0, CONTRAST_MAX, 401),
                _CONTRAST["low"].mf,
                contrast_input,
            )
        ),
        "high_noise": float(
            fuzz.interp_membership(
                np.linspace(0, SNR_MAX, 201),
                _SNR["low"].mf,
                snr_input,
            )
        ),
    }
    dominant_defect = max(defect_memberships, key=defect_memberships.get)
    if defect_memberships[dominant_defect] <= 0:
        dominant_defect = "none"

    return {
        "metrics": {
            "blur": blur_value,
            "contrast": contrast_value,
            "noise": snr_value,
        },
        "fuzzy_score": score,
        "state": state,
        "dominant_defect": dominant_defect,
    }
