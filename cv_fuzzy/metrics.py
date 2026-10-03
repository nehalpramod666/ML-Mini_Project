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
def compute_contrast(img: np.ndarray) -> float:
    """
    Compute contrast via RMS (standard deviation) of pixel intensities.

    Higher values = more contrast/spread.
    Lower values = flat/washed-out image.

    Args:
        img: 2D uint8 grayscale array.

    Returns:
        float RMS deviation of pixel intensities.
    """
    if img.ndim != 2:
        raise ValueError(
            f"compute_contrast expects a 2D grayscale array, "
            f"got shape {img.shape}"
        )

    return float(np.std(img.astype(np.float64)))
def compute_noise(img: np.ndarray) -> float:
    """
    Estimate image noise using a simple SNR metric.

    SNR = mean pixel intensity / standard deviation.

    Higher values indicate a stronger signal relative to variation.
    Lower values indicate more variation/noise.

    Args:
        img: 2D uint8 grayscale array.

    Returns:
        float SNR estimate.

    Raises:
        ValueError: If img is not a 2D grayscale array.
    """
    if img.ndim != 2:
        raise ValueError(
            f"compute_noise expects a 2D grayscale array, "
            f"got shape {img.shape}"
        )

    pixels = img.astype(np.float64)

    mean_value = np.mean(pixels)
    std_value = np.std(pixels)

    # Avoid division by zero for a completely flat image.
    if std_value == 0:
        return float("inf")

    return float(mean_value / std_value)
def analyze_image(img) -> dict:
    return {
        "metrics": {
            "blur": compute_blur(img),
            "contrast": compute_contrast(img),
            "noise": compute_noise(img),
        }
    }