
"""Experimental salt-and-pepper impulse-noise estimator.

This module is for evaluation only. It does not replace the
production SNR estimator or change fuzzy-state thresholds.
"""

from numbers import Integral, Real

import cv2
import numpy as np


def estimate_impulse_noise(
    img: np.ndarray,
    *,
    kernel_size: int = 3,
    residual_threshold: float = 64.0,
) -> dict:
    """Estimate the fraction of isolated black/white impulse pixels.

    A candidate pixel must:
      1. Have intensity 0 or 255.
      2. Differ from its local median by at least
         residual_threshold intensity levels.

    Args:
        img: Two-dimensional grayscale image with intensities in [0, 255].
        kernel_size: Odd median-filter size; defaults to 3.
        residual_threshold: Minimum absolute residual; defaults to 64.

    Returns:
        Dictionary containing pixel counts and impulse density.

    Raises:
        TypeError: If the image or parameters have invalid types.
        ValueError: If the image or parameters are invalid, or the
            image contains NaN or infinity.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")

    if img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    if img.size == 0:
        raise ValueError("img must not be empty")

    if not np.issubdtype(img.dtype, np.number):
        raise TypeError("Image pixels must have a numeric dtype")

    if not np.isfinite(img).all():
        raise ValueError("Image must contain only finite pixel values")

    if np.any(img < 0) or np.any(img > 255):
        raise ValueError("Image intensities must be in [0, 255]")

    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, Integral)
        or kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be an odd integer >= 3")

    if (
        isinstance(residual_threshold, bool)
        or not isinstance(residual_threshold, Real)
        or not np.isfinite(residual_threshold)
        or not 0 < residual_threshold <= 255
    ):
        raise ValueError(
            "residual_threshold must be finite and in (0, 255]"
        )

    # The source pipeline uses normalized 8-bit grayscale images.
    image = np.rint(img).astype(np.uint8)

    local_median = cv2.medianBlur(image, int(kernel_size))

    # Convert before subtraction to avoid unsigned integer wraparound.
    residual = np.abs(
        image.astype(np.int16)
        - local_median.astype(np.int16)
    )

    extreme_pixel = (image == 0) | (image == 255)

    impulse_mask = (
        extreme_pixel
        & (residual >= residual_threshold)
    )

    impulse_pixels = int(np.count_nonzero(impulse_mask))
    pixel_count = int(image.size)
    impulse_fraction = impulse_pixels / pixel_count

    return {
        "impulse_pixels": impulse_pixels,
        "pixel_count": pixel_count,
        "impulse_fraction": float(impulse_fraction),
        "impulse_percent": float(100.0 * impulse_fraction),
        "kernel_size": int(kernel_size),
        "residual_threshold": float(residual_threshold),
    }
