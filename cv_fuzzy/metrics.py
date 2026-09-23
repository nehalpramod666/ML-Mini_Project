"""
Classical image quality metrics: blur, contrast, noise.
"""
import cv2
import numpy as np


def compute_blur(img: np.ndarray) -> float:
    """
    Compute a blur score via variance of the Laplacian.

    Higher values = sharper image. Lower values = blurrier image.

    Args:
        img: 2D uint8 grayscale array (output of load_dicom).

    Returns:
        float variance of the Laplacian filter response.
    """
    if img.ndim != 2:
        raise ValueError(f"compute_blur expects a 2D grayscale array, got shape {img.shape}")

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(laplacian.var())