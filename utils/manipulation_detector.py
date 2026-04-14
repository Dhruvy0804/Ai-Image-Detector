"""
Manipulation Detector Module
=============================
Detects which specific regions of an image have been AI-modified
vs. original camera-captured content.

Uses multiple forensic techniques:
  1. Error Level Analysis (ELA)
  2. Noise Variance Map
  3. Local Frequency Anomaly Detection
  4. Combined Manipulation Heatmap

When an AI tool edits a real photo (e.g., changing clothes, adding objects),
the modified regions leave detectable forensic traces that differ from the
original camera-captured areas.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageChops
from scipy import ndimage, fftpack
import cv2
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
# 1. Error Level Analysis (ELA)
# ============================================================

def compute_ela(img, quality=90, scale=15):
    """
    Error Level Analysis — detects regions with inconsistent
    JPEG compression error levels.

    When an AI edits part of an image and re-saves it, the edited
    regions have different compression artifacts than the original
    regions. ELA amplifies these differences.

    Args:
        img: PIL Image
        quality: JPEG quality to re-save at (lower = more sensitive)
        scale: Amplification factor for the error map

    Returns:
        ela_image: PIL Image of the amplified error
        ela_array: numpy array (H, W) normalized 0-1
    """
    # Ensure RGB
    img_rgb = img.convert('RGB')

    # Re-save at known quality
    buffer = io.BytesIO()
    img_rgb.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert('RGB')

    # Compute pixel-level difference
    ela_image = ImageChops.difference(img_rgb, resaved)

    # Convert to numpy and amplify
    ela_array = np.array(ela_image, dtype=np.float32)
    ela_array = ela_array * scale

    # Clip to 0-255
    ela_array = np.clip(ela_array, 0, 255)

    # Convert to grayscale intensity
    ela_gray = np.mean(ela_array, axis=2)

    # Normalize to 0-1
    if ela_gray.max() > 0:
        ela_normalized = ela_gray / ela_gray.max()
    else:
        ela_normalized = ela_gray

    # Apply Gaussian blur to smooth the map
    ela_smoothed = ndimage.gaussian_filter(ela_normalized, sigma=3)

    # Re-normalize after smoothing
    if ela_smoothed.max() > 0:
        ela_smoothed = ela_smoothed / ela_smoothed.max()

    # Create colored ELA image for display
    ela_display = Image.fromarray(np.uint8(ela_array))

    return ela_display, ela_smoothed


def compute_ela_multi_quality(img, qualities=[95, 90, 85, 75]):
    """
    Run ELA at multiple quality levels and combine results.
    More robust than single-quality ELA.
    """
    combined = None
    for q in qualities:
        _, ela_map = compute_ela(img, quality=q, scale=20)
        if combined is None:
            combined = ela_map
        else:
            combined = np.maximum(combined, ela_map)

    if combined is not None and combined.max() > 0:
        combined = combined / combined.max()

    return combined


# ============================================================
# 2. Noise Variance Analysis
# ============================================================

def compute_noise_variance_map(img, block_size=16):
    """
    Compute local noise variance across the image.

    Camera-captured regions have consistent noise from the sensor.
    AI-generated/edited regions have different noise characteristics
    (often smoother or with different texture patterns).

    Regions with significantly different noise variance from the
    image average are flagged as potentially manipulated.

    Args:
        img: PIL Image
        block_size: Size of analysis blocks

    Returns:
        noise_map: numpy array (H, W) normalized 0-1, higher = more anomalous
    """
    # Convert to grayscale numpy array
    img_gray = np.array(img.convert('L'), dtype=np.float64)

    h, w = img_gray.shape

    # Apply high-pass filter to isolate noise
    # Using Laplacian which responds to rapid intensity changes
    noise_residual = ndimage.laplace(img_gray)

    # Compute local variance in blocks
    pad_h = block_size - (h % block_size) if h % block_size != 0 else 0
    pad_w = block_size - (w % block_size) if w % block_size != 0 else 0
    noise_padded = np.pad(noise_residual, ((0, pad_h), (0, pad_w)), mode='reflect')

    new_h, new_w = noise_padded.shape
    blocks_h = new_h // block_size
    blocks_w = new_w // block_size

    variance_map = np.zeros((blocks_h, blocks_w))

    for i in range(blocks_h):
        for j in range(blocks_w):
            block = noise_padded[
                i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size
            ]
            variance_map[i, j] = np.var(block)

    # Compute global median variance
    median_var = np.median(variance_map)

    # Anomaly = deviation from median (both too smooth and too noisy)
    if median_var > 0:
        anomaly_map = np.abs(variance_map - median_var) / median_var
    else:
        anomaly_map = np.zeros_like(variance_map)

    # Resize back to original image size
    anomaly_resized = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize
    if anomaly_resized.max() > 0:
        anomaly_resized = anomaly_resized / anomaly_resized.max()

    # Smooth
    anomaly_smoothed = ndimage.gaussian_filter(anomaly_resized, sigma=5)
    if anomaly_smoothed.max() > 0:
        anomaly_smoothed = anomaly_smoothed / anomaly_smoothed.max()

    return anomaly_smoothed


def compute_noise_inconsistency(img):
    """
    Detect noise inconsistency using wavelet-like decomposition.
    Uses multiple scales for more robust detection.
    """
    img_gray = np.array(img.convert('L'), dtype=np.float64)

    # Multi-scale noise analysis
    scales = [3, 5, 7]
    combined = np.zeros_like(img_gray)

    for s in scales:
        # Denoise at this scale
        denoised = ndimage.median_filter(img_gray, size=s)
        # Noise residual
        residual = np.abs(img_gray - denoised)
        # Local variance of residual
        local_var = ndimage.uniform_filter(residual ** 2, size=16)
        local_mean = ndimage.uniform_filter(residual, size=16) ** 2
        local_std = np.sqrt(np.maximum(local_var - local_mean, 0))

        combined += local_std

    combined = combined / len(scales)

    # Detect inconsistencies (deviation from image median)
    median_val = np.median(combined)
    if median_val > 0:
        inconsistency = np.abs(combined - median_val) / median_val
    else:
        inconsistency = np.zeros_like(combined)

    # Normalize and smooth
    if inconsistency.max() > 0:
        inconsistency = inconsistency / inconsistency.max()

    inconsistency = ndimage.gaussian_filter(inconsistency, sigma=4)
    if inconsistency.max() > 0:
        inconsistency = inconsistency / inconsistency.max()

    return inconsistency


# ============================================================
# 3. Local Frequency Anomaly Detection
# ============================================================

def compute_local_frequency_map(img, window_size=64, stride=16):
    """
    Sliding-window FFT analysis to detect regions with different
    spectral characteristics.

    AI-generated textures have distinct frequency signatures compared
    to camera-captured textures. This function analyzes local patches
    and flags those that differ from the image's dominant spectral pattern.

    Args:
        img: PIL Image
        window_size: Size of the analysis window
        stride: Step size between windows

    Returns:
        freq_anomaly_map: numpy array (H, W) normalized 0-1
    """
    img_gray = np.array(img.convert('L'), dtype=np.float64)
    h, w = img_gray.shape

    # Adjust window size if image is smaller
    window_size = min(window_size, min(h, w))
    if window_size < 16:
        return np.zeros((h, w))

    stride = min(stride, window_size // 2)

    # Compute global spectral reference
    global_fft = fftpack.fft2(img_gray)
    global_magnitude = np.log1p(np.abs(fftpack.fftshift(global_fft)))
    global_high_ratio = _compute_high_freq_ratio(global_magnitude)

    # Sliding window analysis
    rows = list(range(0, h - window_size + 1, stride))
    cols = list(range(0, w - window_size + 1, stride))

    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((h, w))

    anomaly_grid = np.zeros((len(rows), len(cols)))

    local_ratios = []

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            patch = img_gray[r:r + window_size, c:c + window_size]

            # Local FFT
            local_fft = fftpack.fft2(patch)
            local_magnitude = np.log1p(np.abs(fftpack.fftshift(local_fft)))

            local_ratio = _compute_high_freq_ratio(local_magnitude)
            local_ratios.append(local_ratio)
            anomaly_grid[i, j] = local_ratio

    # Compute deviation from median local ratio
    median_ratio = np.median(local_ratios)
    if median_ratio > 0:
        anomaly_grid = np.abs(anomaly_grid - median_ratio) / (median_ratio + 1e-10)
    else:
        anomaly_grid = np.zeros_like(anomaly_grid)

    # Resize to original image size
    freq_map = cv2.resize(anomaly_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize
    if freq_map.max() > 0:
        freq_map = freq_map / freq_map.max()

    # Smooth
    freq_map = ndimage.gaussian_filter(freq_map, sigma=6)
    if freq_map.max() > 0:
        freq_map = freq_map / freq_map.max()

    return freq_map


def _compute_high_freq_ratio(magnitude_spectrum):
    """Compute the ratio of high-frequency energy to total energy."""
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
    distances = np.sqrt(x ** 2 + y ** 2)
    max_r = min(cx, cy)

    high_mask = distances >= max_r * 0.5
    total = np.sum(magnitude_spectrum)

    if total == 0:
        return 0.0

    return float(np.sum(magnitude_spectrum[high_mask]) / total)


# ============================================================
# 4. Edge Coherence Analysis
# ============================================================

def compute_edge_inconsistency(img):
    """
    Detect boundary artifacts between real and AI-edited regions.

    AI edits often create subtle edge artifacts at the boundary
    between the original and modified areas — slightly blurred
    transitions, unnatural edge directions, or discontinuities.

    Returns:
        edge_map: numpy array (H, W) normalized 0-1
    """
    img_gray = np.array(img.convert('L'), dtype=np.float64)

    # Compute edges using multiple methods
    # Sobel in x and y
    sobel_x = ndimage.sobel(img_gray, axis=1)
    sobel_y = ndimage.sobel(img_gray, axis=0)
    edge_magnitude = np.hypot(sobel_x, sobel_y)

    # Compute local edge density variance
    edge_density = ndimage.uniform_filter(edge_magnitude, size=32)
    edge_density_var = ndimage.uniform_filter(edge_magnitude ** 2, size=32)
    edge_local_var = edge_density_var - edge_density ** 2
    edge_local_var = np.maximum(edge_local_var, 0)

    # Detect transitions — areas where edge characteristics change abruptly
    edge_gradient = np.hypot(
        ndimage.sobel(edge_density, axis=0),
        ndimage.sobel(edge_density, axis=1)
    )

    # Normalize
    if edge_gradient.max() > 0:
        edge_gradient = edge_gradient / edge_gradient.max()

    # Smooth
    edge_map = ndimage.gaussian_filter(edge_gradient, sigma=5)
    if edge_map.max() > 0:
        edge_map = edge_map / edge_map.max()

    return edge_map


# ============================================================
# 5. Combined Manipulation Map
# ============================================================

def compute_manipulation_map(img, sensitivity=0.5, use_ela=True,
                             use_noise=True, use_frequency=True,
                             use_edge=True):
    """
    Combine all forensic techniques into a single manipulation map.

    Args:
        img: PIL Image
        sensitivity: 0.0 (low) to 1.0 (high) — controls detection threshold
        use_ela: Whether to include ELA analysis
        use_noise: Whether to include noise variance analysis
        use_frequency: Whether to include local frequency analysis
        use_edge: Whether to include edge coherence analysis

    Returns:
        combined_map: numpy array (H, W) normalized 0-1 (higher = more likely AI-edited)
        individual_maps: dict of individual analysis maps
        stats: dict of statistics about the detection
    """
    h, w = np.array(img.convert('L')).shape
    maps = {}
    weights = {}

    # Run enabled analyses
    if use_ela:
        _, ela_map = compute_ela(img, quality=90, scale=15)
        ela_multi = compute_ela_multi_quality(img)
        # Average single and multi-quality ELA
        maps['ela'] = (ela_map + ela_multi) / 2.0 if ela_multi is not None else ela_map
        weights['ela'] = 0.35

    if use_noise:
        noise_map = compute_noise_variance_map(img, block_size=16)
        noise_inconsistency = compute_noise_inconsistency(img)
        maps['noise'] = (noise_map + noise_inconsistency) / 2.0
        weights['noise'] = 0.25

    if use_frequency:
        freq_map = compute_local_frequency_map(img, window_size=64, stride=16)
        maps['frequency'] = freq_map
        weights['frequency'] = 0.25

    if use_edge:
        edge_map = compute_edge_inconsistency(img)
        maps['edge'] = edge_map
        weights['edge'] = 0.15

    if not maps:
        return np.zeros((h, w)), {}, {}

    # Normalize weights
    total_weight = sum(weights.values())
    for k in weights:
        weights[k] /= total_weight

    # Weighted combination
    combined = np.zeros((h, w))
    for key, m in maps.items():
        # Resize map to match image dimensions if needed
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            maps[key] = m
        combined += weights[key] * m

    # Normalize combined map
    if combined.max() > 0:
        combined = combined / combined.max()

    # Apply sensitivity threshold
    # Higher sensitivity = lower threshold = more areas flagged
    threshold = 0.5 * (1.0 - sensitivity)
    combined_thresholded = np.where(combined > threshold, combined, combined * 0.3)

    # Re-normalize
    if combined_thresholded.max() > 0:
        combined_thresholded = combined_thresholded / combined_thresholded.max()

    # Compute statistics
    manipulation_percentage = float(np.mean(combined_thresholded > 0.5) * 100)
    max_intensity = float(np.max(combined_thresholded))
    mean_intensity = float(np.mean(combined_thresholded))

    stats = {
        'manipulation_percentage': manipulation_percentage,
        'max_intensity': max_intensity,
        'mean_intensity': mean_intensity,
        'weights': weights,
        'num_techniques': len(maps),
    }

    return combined_thresholded, maps, stats


# ============================================================
# 6. Direct Comparison (with original)
# ============================================================

def compute_direct_diff(original_img, edited_img, amplification=5):
    """
    When the user provides both the original and AI-edited image,
    compute a direct pixel-level comparison.

    This gives the most accurate manipulation map possible.

    Args:
        original_img: PIL Image (original photo)
        edited_img: PIL Image (AI-edited version)
        amplification: How much to amplify differences

    Returns:
        diff_map: numpy array (H, W) normalized 0-1
        diff_color: PIL Image showing colored differences
    """
    # Ensure same size
    orig = original_img.convert('RGB')
    edit = edited_img.convert('RGB')

    # Resize edited to match original if needed
    if orig.size != edit.size:
        edit = edit.resize(orig.size, Image.LANCZOS)

    orig_arr = np.array(orig, dtype=np.float64)
    edit_arr = np.array(edit, dtype=np.float64)

    # Compute per-channel difference
    diff = np.abs(orig_arr - edit_arr)

    # Amplify
    diff = diff * amplification
    diff = np.clip(diff, 0, 255)

    # Convert to grayscale intensity for map
    diff_gray = np.mean(diff, axis=2)

    # Normalize
    if diff_gray.max() > 0:
        diff_normalized = diff_gray / diff_gray.max()
    else:
        diff_normalized = diff_gray

    # Smooth slightly
    diff_smoothed = ndimage.gaussian_filter(diff_normalized, sigma=2)
    if diff_smoothed.max() > 0:
        diff_smoothed = diff_smoothed / diff_smoothed.max()

    # Create colored diff image
    diff_color = Image.fromarray(np.uint8(diff))

    return diff_smoothed, diff_color


# ============================================================
# 7. Visualization Functions
# ============================================================

def create_heatmap_overlay(img, manipulation_map, alpha=0.5, colormap='RdYlBu_r'):
    """
    Overlay the manipulation heatmap on the original image.

    Red = likely AI-edited regions
    Blue = likely original/real regions

    Args:
        img: PIL Image
        manipulation_map: numpy array (H, W) 0-1
        alpha: overlay transparency
        colormap: matplotlib colormap name

    Returns:
        overlaid: numpy array (H, W, 3) uint8
    """
    img_rgb = np.array(img.convert('RGB'), dtype=np.float64) / 255.0
    h, w = img_rgb.shape[:2]

    # Resize map if needed
    if manipulation_map.shape != (h, w):
        manipulation_map = cv2.resize(manipulation_map, (w, h),
                                       interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(manipulation_map)[:, :, :3]  # Remove alpha channel

    # Blend
    overlaid = (1 - alpha) * img_rgb + alpha * heatmap_colored
    overlaid = np.clip(overlaid * 255, 0, 255).astype(np.uint8)

    return overlaid


def create_manipulation_figure(img, manipulation_map, individual_maps,
                                stats, label="Manipulation Analysis"):
    """
    Create a comprehensive matplotlib figure showing the manipulation analysis.
    """
    n_maps = 1 + len(individual_maps)  # combined + individual
    n_cols = min(3, n_maps + 1)  # +1 for original
    n_rows = (n_maps + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor('#0F0F1A')

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    flat_axes = axes.flatten()

    for ax in flat_axes:
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')

    # 1. Original image
    img_rgb = np.array(img.convert('RGB'))
    flat_axes[0].imshow(img_rgb)
    flat_axes[0].set_title('Original Image', color='#E2E8F0',
                            fontsize=12, fontweight='bold')

    # 2. Combined manipulation map
    overlay = create_heatmap_overlay(img, manipulation_map, alpha=0.55)
    flat_axes[1].imshow(overlay)
    pct = stats.get('manipulation_percentage', 0)
    flat_axes[1].set_title(f'Combined Map ({pct:.1f}% flagged)',
                            color='#E2E8F0', fontsize=12, fontweight='bold')

    # 3+ Individual maps
    map_names = {
        'ela': ('ELA (Error Levels)', 'hot'),
        'noise': ('Noise Inconsistency', 'inferno'),
        'frequency': ('Frequency Anomaly', 'magma'),
        'edge': ('Edge Coherence', 'plasma'),
    }

    idx = 2
    for key, m in individual_maps.items():
        if idx < len(flat_axes):
            name, cmap = map_names.get(key, (key, 'viridis'))
            flat_axes[idx].imshow(m, cmap=cmap, vmin=0, vmax=1)
            flat_axes[idx].set_title(name, color='#E2E8F0',
                                      fontsize=12, fontweight='bold')
            idx += 1

    # Hide unused axes
    for i in range(idx, len(flat_axes)):
        flat_axes[i].set_visible(False)

    plt.tight_layout(pad=2.0)
    return fig


def plot_individual_map(img, analysis_map, title, cmap='hot'):
    """
    Plot a single analysis map with overlay on the original image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0F0F1A')

    for ax in axes:
        ax.set_facecolor('#1A1A2E')
        ax.axis('off')

    img_rgb = np.array(img.convert('RGB'))

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original', color='#E2E8F0', fontsize=12, fontweight='bold')

    # Heatmap only
    im = axes[1].imshow(analysis_map, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f'{title} Map', color='#E2E8F0', fontsize=12, fontweight='bold')

    # Overlay
    overlay = create_heatmap_overlay(img, analysis_map, alpha=0.5, colormap=cmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f'{title} Overlay', color='#E2E8F0', fontsize=12, fontweight='bold')

    plt.tight_layout(pad=2.0)
    return fig
