"""
Error Level Analysis (ELA) Module
==================================
Detects image tampering by analysing JPEG re-compression inconsistencies.
Authentic images have uniform error levels; tampered regions show higher errors.

Author: Amit Mishra
"""

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import os
import tempfile


def generate_ela(image_path, quality=90, scale=10):
    """
    Generate Error Level Analysis image.

    Args:
        image_path (str): Path to input image
        quality (int): JPEG quality for re-compression (default: 90)
        scale (int): Amplification scale for visualisation (default: 10)

    Returns:
        ela_array (np.ndarray): ELA image as numpy array
        ela_pil (PIL.Image): ELA image as PIL Image
    """
    # Load original image
    original = Image.open(image_path).convert('RGB')

    # Save at reduced quality to temp file
    temp_path = os.path.join(tempfile.gettempdir(), 'ela_temp.jpg')
    original.save(temp_path, 'JPEG', quality=quality)

    # Reload re-compressed image
    recompressed = Image.open(temp_path).convert('RGB')

    # Compute pixel-wise absolute difference
    ela_image = ImageChops.difference(original, recompressed)

    # Scale differences for visibility
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale_factor = 255.0 / max_diff * scale
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale_factor)

    ela_array = np.array(ela_image)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return ela_array, ela_image


def ela_heatmap(image_path, quality=90):
    """
    Generate a single-channel ELA heatmap (grayscale).

    Args:
        image_path (str): Path to input image
        quality (int): JPEG quality

    Returns:
        heatmap (np.ndarray): Single channel heatmap
    """
    ela_array, _ = generate_ela(image_path, quality=quality)
    heatmap = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
    return heatmap


def compute_ela_statistics(image_path, quality=90):
    """
    Compute statistical features from ELA image for ML classification.

    Args:
        image_path (str): Path to input image
        quality (int): JPEG quality

    Returns:
        features (dict): Statistical features from ELA
    """
    ela_array, _ = generate_ela(image_path, quality=quality)
    gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)

    features = {
        'ela_mean':     float(np.mean(gray)),
        'ela_std':      float(np.std(gray)),
        'ela_max':      float(np.max(gray)),
        'ela_min':      float(np.min(gray)),
        'ela_median':   float(np.median(gray)),
        'ela_skewness': float(_skewness(gray.flatten())),
        'ela_kurtosis': float(_kurtosis(gray.flatten())),
        # Regional statistics (divide image into 4 quadrants)
        'ela_q1_mean':  float(np.mean(gray[:gray.shape[0]//2, :gray.shape[1]//2])),
        'ela_q2_mean':  float(np.mean(gray[:gray.shape[0]//2, gray.shape[1]//2:])),
        'ela_q3_mean':  float(np.mean(gray[gray.shape[0]//2:, :gray.shape[1]//2])),
        'ela_q4_mean':  float(np.mean(gray[gray.shape[0]//2:, gray.shape[1]//2:])),
    }
    return features


def visualise_ela(image_path, quality=90, save_path=None):
    """
    Visualise original image alongside its ELA output.

    Args:
        image_path (str): Path to input image
        quality (int): JPEG quality
        save_path (str): Optional path to save the visualisation
    """
    original = Image.open(image_path).convert('RGB')
    ela_array, ela_pil = generate_ela(image_path, quality=quality)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(ela_array)
    axes[1].set_title(f'Error Level Analysis (Quality={quality})',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle('ELA Forensic Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"ELA visualisation saved to: {save_path}")

    plt.show()
    return fig


def multi_quality_ela(image_path, qualities=[70, 80, 90, 95]):
    """
    Generate ELA at multiple JPEG quality levels for robustness analysis.

    Args:
        image_path (str): Path to input image
        qualities (list): List of JPEG quality values

    Returns:
        results (dict): ELA arrays at each quality level
    """
    results = {}
    fig, axes = plt.subplots(1, len(qualities), figsize=(5 * len(qualities), 5))

    for i, q in enumerate(qualities):
        ela_array, _ = generate_ela(image_path, quality=q)
        results[q] = ela_array
        axes[i].imshow(ela_array)
        axes[i].set_title(f'Quality={q}', fontsize=12)
        axes[i].axis('off')

    plt.suptitle('ELA at Multiple Quality Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return results


# ── Helper functions ─────────────────────────────────────────────────────────

def _skewness(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 3))


def _kurtosis(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(np.mean(((data - mean) / std) ** 4) - 3)


# ── Main demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python ela.py <image_path>")
        print("Example: python ela.py data/sample_images/test.jpg")
        sys.exit(0)

    print(f"\n[ELA] Analysing: {image_path}")
    print("-" * 50)

    # ELA statistics
    stats = compute_ela_statistics(image_path)
    print("ELA Statistical Features:")
    for key, val in stats.items():
        print(f"  {key:20s}: {val:.4f}")

    # Visualise
    visualise_ela(image_path, save_path="results/ela_output.png")
    print("\n[ELA] Analysis complete.")
