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


def generate_gold_standard_set(clean_image_paths, output_dir, n_per_defect=33):
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    defect_fns = {
        "blur": lambda img: add_gaussian_blur(
            img, kernel_size=random.choice([9, 15, 21])
        ),
        "noise": lambda img: add_salt_pepper_noise(
            img, amount=random.choice([0.01, 0.02, 0.05])
        ),
        "contrast": lambda img: reduce_contrast(
            img, factor=random.choice([0.3, 0.5, 0.7])
        ),
    }

    idx = 0

    for defect_name, fn in defect_fns.items():
        for i in range(n_per_defect):
            src_path = random.choice(clean_image_paths)

            img = load_dicom(src_path)
            degraded = fn(img)

            out_name = f"{defect_name}_{idx:03d}.png"
            cv2.imwrite(
                os.path.join(output_dir, out_name),
                degraded,
            )

            manifest.append({
                "filename": out_name,
                "injected_defect": defect_name,
                "source": src_path,
            })

            idx += 1

    with open(
        os.path.join(output_dir, "manifest.json"),
        "w",
    ) as f:
        json.dump(manifest, f, indent=2)

    return manifest