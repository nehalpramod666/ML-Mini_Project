"""
Synthetic image degradation utilities for M2 validation.

Generates controlled blur, noise, and contrast defects from
clean 2D grayscale images.
"""

import cv2
import numpy as np
import json
import os
import random

from cv_fuzzy.dicom_loader import load_dicom


def add_gaussian_blur(
    img: np.ndarray,
    kernel_size: int = 15,
    sigma: float = 5,
) -> np.ndarray:
    """Simulate out-of-focus blur."""
    if img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    return cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        sigmaX=sigma,
    )


def add_salt_pepper_noise(
    img: np.ndarray,
    amount: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Add randomly distributed black and white pixels."""
    if img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    if not 0 <= amount <= 1:
        raise ValueError("amount must be between 0 and 1")

    noisy = img.copy()
    rng = np.random.default_rng(seed)

    num_pixels = img.size
    num_salt = int(amount * num_pixels * 0.5)
    num_pepper = int(amount * num_pixels * 0.5)

    salt_indices = rng.choice(
        num_pixels, size=num_salt, replace=False
    )
    pepper_indices = rng.choice(
        num_pixels, size=num_pepper, replace=False
    )

    noisy.flat[salt_indices] = 255
    noisy.flat[pepper_indices] = 0

    return noisy


def reduce_contrast(
    img: np.ndarray,
    factor: float = 0.5,
) -> np.ndarray:
    """Reduce the intensity spread around the image mean."""
    if img.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    if not 0 <= factor <= 1:
        raise ValueError("factor must be between 0 and 1")

    mean = img.mean()

    return np.clip(
        (img.astype(np.float64) - mean) * factor + mean,
        0,
        255,
    ).astype(np.uint8)



def generate_gold_standard_set(
    clean_image_paths,
    output_dir,
    seed=42,
):
    """Generate 100 reproducible degraded images and a JSON manifest."""
    if not clean_image_paths:
        raise ValueError("clean_image_paths must not be empty")

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(seed)
    manifest = []

    defect_counts = {
        "blur": 34,
        "noise": 33,
        "contrast": 33,
    }

    idx = 0

    for defect_name, count in defect_counts.items():
        for _ in range(count):
            src_path = rng.choice(clean_image_paths)
            img = load_dicom(src_path)

            if img.ndim != 2:
                raise ValueError(
                    f"Expected a 2D grayscale image: {src_path}"
                )

            if defect_name == "blur":
                kernel_size = rng.choice([9, 15, 21])
                sigma = 5.0

                degraded = add_gaussian_blur(
                    img,
                    kernel_size=kernel_size,
                    sigma=sigma,
                )

                parameters = {
                    "kernel_size": kernel_size,
                    "sigma": sigma,
                }

            elif defect_name == "noise":
                amount = rng.choice([0.01, 0.02, 0.05])
                noise_seed = rng.randrange(0, 2**32)

                degraded = add_salt_pepper_noise(
                    img,
                    amount=amount,
                    seed=noise_seed,
                )

                parameters = {
                    "amount": amount,
                    "seed": noise_seed,
                }

            else:
                factor = rng.choice([0.3, 0.5, 0.7])

                degraded = reduce_contrast(
                    img,
                    factor=factor,
                )

                parameters = {
                    "factor": factor,
                }

            out_name = f"{defect_name}_{idx:03d}.png"
            out_path = os.path.join(output_dir, out_name)

            if not cv2.imwrite(out_path, degraded):
                raise RuntimeError(
                    f"Failed to save image: {out_path}"
                )

            manifest.append({
                "filename": out_name,
                "injected_defect": defect_name,
                "source": str(src_path),
                "parameters": parameters,
                "seed": seed,
            })

            idx += 1

    manifest_path = os.path.join(output_dir, "manifest.json")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
