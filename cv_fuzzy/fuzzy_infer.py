"""M2 orchestration for generating a validated V2 image-quality payload.

This is the confirmed M2 entry point for grayscale NumPy images. It composes
the existing fuzzy engine and V2 contract builder.
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
