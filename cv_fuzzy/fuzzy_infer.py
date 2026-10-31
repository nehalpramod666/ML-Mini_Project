"""Provisional M2 orchestration for generating a V2 image-quality payload.

This module composes existing M2 components. Its public integration
interface remains provisional until M1 and M3 confirm their requirements.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .fuzzy_engine import analyze_image_fuzzy
from .v2_contract import build_v2_payload


def fuzzy_infer(image: np.ndarray) -> dict[str, Any]:
    """Generate a validated V2 payload from a grayscale NumPy image.

    Image validation and error semantics are delegated to the existing
    fuzzy engine and V2 contract builder.

    Exceptions are deliberately not caught or translated. In particular,
    FuzzyInferenceError and ContractV2Error retain their original types,
    messages, and error codes.
    """
    m2_result = analyze_image_fuzzy(image)
    return build_v2_payload(m2_result, image)
