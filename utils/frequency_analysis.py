"""
Frequency Analysis Module
Performs FFT-based frequency domain analysis to detect
artifacts commonly found in AI-generated images.
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io


def compute_fft_spectrum(image_gray):
    """
    Compute the 2D FFT magnitude spectrum of a grayscale image.
    
    AI-generated images often show:
    - Grid-like patterns in frequency domain
    - Unusual energy distribution  
    - Periodic artifacts from the generation process
    
    Args:
        image_gray: 2D numpy array (grayscale image)
    
    Returns:
        magnitude_spectrum: Log-scaled magnitude spectrum
        phase_spectrum: Phase spectrum
    """
    # Apply 2D FFT
    f_transform = fftpack.fft2(image_gray)
    
    # Shift zero frequency to center
    f_shift = fftpack.fftshift(f_transform)
    
    # Compute magnitude and phase spectra
    magnitude_spectrum = np.log1p(np.abs(f_shift))
    phase_spectrum = np.angle(f_shift)
    
    return magnitude_spectrum, phase_spectrum


def compute_azimuthal_average(spectrum):
    """
    Compute the azimuthally averaged power spectrum.
    This reveals the radial frequency distribution which
    differs between real and AI-generated images.
    """
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Create distance matrix from center
    y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    distances = np.sqrt(x**2 + y**2).astype(int)
    
    max_radius = min(center_x, center_y)
    
    # Compute average at each radius
    radial_profile = np.zeros(max_radius)
    for r in range(max_radius):
        mask = distances == r
        if np.any(mask):
            radial_profile[r] = np.mean(spectrum[mask])
    
    return radial_profile


def compute_spectral_features(image_gray):
    """
    Extract frequency-domain features for AI detection.
    
    Returns a dict of spectral features and anomaly indicators.
    """
    magnitude_spectrum, _ = compute_fft_spectrum(image_gray)
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Divide spectrum into low, mid, high frequency bands
    y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    distances = np.sqrt(x**2 + y**2)
    max_radius = min(center_x, center_y)
    
    low_mask = distances < max_radius * 0.2
    mid_mask = (distances >= max_radius * 0.2) & (distances < max_radius * 0.6)
    high_mask = distances >= max_radius * 0.6
    
    total_energy = np.sum(magnitude_spectrum)
    
    if total_energy == 0:
        total_energy = 1e-10
    
    low_energy = np.sum(magnitude_spectrum[low_mask]) / total_energy
    mid_energy = np.sum(magnitude_spectrum[mid_mask]) / total_energy
    high_energy = np.sum(magnitude_spectrum[high_mask]) / total_energy
    
    # Compute spectral entropy (uniformity of spectrum)
    norm_spectrum = magnitude_spectrum / total_energy
    norm_spectrum = norm_spectrum[norm_spectrum > 0]
    spectral_entropy = -np.sum(norm_spectrum * np.log2(norm_spectrum + 1e-10))
    
    # Compute radial profile
    radial_profile = compute_azimuthal_average(magnitude_spectrum)
    
    # Spectral slope (how fast energy drops with frequency)
    if len(radial_profile) > 1:
        x_vals = np.arange(1, len(radial_profile) + 1)
        log_x = np.log(x_vals + 1e-10)
        log_y = np.log(radial_profile + 1e-10)
        
        # Linear regression for spectral slope
        valid = np.isfinite(log_x) & np.isfinite(log_y)
        if np.sum(valid) > 2:
            coeffs = np.polyfit(log_x[valid], log_y[valid], 1)
            spectral_slope = coeffs[0]
        else:
            spectral_slope = 0.0
    else:
        spectral_slope = 0.0
    
    # Anomaly scoring
    # AI images tend to have: more uniform spectral distribution,
    # higher high-frequency energy, steeper or unusual spectral slopes
    anomaly_indicators = []
    
    if high_energy > 0.35:
        anomaly_indicators.append("High-frequency energy elevated")
    if abs(spectral_slope) < 1.0:
        anomaly_indicators.append("Unusually flat spectral slope")
    if spectral_entropy > 15.0:
        anomaly_indicators.append("High spectral entropy (uniform distribution)")
    
    # Simple anomaly score (0 = likely real, 1 = likely AI)
    anomaly_score = 0.0
    anomaly_score += min(high_energy / 0.4, 1.0) * 0.3
    anomaly_score += max(0, 1 - abs(spectral_slope) / 2.0) * 0.3
    anomaly_score += min(spectral_entropy / 20.0, 1.0) * 0.4
    anomaly_score = np.clip(anomaly_score, 0, 1)
    
    return {
        "low_frequency_ratio": float(low_energy),
        "mid_frequency_ratio": float(mid_energy),
        "high_frequency_ratio": float(high_energy),
        "spectral_entropy": float(spectral_entropy),
        "spectral_slope": float(spectral_slope),
        "anomaly_score": float(anomaly_score),
        "anomaly_indicators": anomaly_indicators,
        "radial_profile": radial_profile,
    }


def plot_frequency_spectrum(image_gray, figsize=(12, 4)):
    """
    Generate a visualization of the frequency analysis.
    Returns a matplotlib figure.
    """
    magnitude_spectrum, phase_spectrum = compute_fft_spectrum(image_gray)
    features = compute_spectral_features(image_gray)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('#0F0F1A')
    
    for ax in axes:
        ax.set_facecolor('#1A1A2E')
        ax.tick_params(colors='#94A3B8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#334155')
    
    # 1. Magnitude Spectrum
    im1 = axes[0].imshow(magnitude_spectrum, cmap='inferno', aspect='auto')
    axes[0].set_title('Magnitude Spectrum', color='#E2E8F0', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Phase Spectrum
    im2 = axes[1].imshow(phase_spectrum, cmap='twilight', aspect='auto')
    axes[1].set_title('Phase Spectrum', color='#E2E8F0', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Radial Power Profile
    radial = features['radial_profile']
    axes[2].plot(radial, color='#8B5CF6', linewidth=2, alpha=0.9)
    axes[2].fill_between(range(len(radial)), radial, alpha=0.15, color='#8B5CF6')
    axes[2].set_title('Radial Power Profile', color='#E2E8F0', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Frequency', color='#94A3B8', fontsize=9)
    axes[2].set_ylabel('Power', color='#94A3B8', fontsize=9)
    axes[2].grid(True, alpha=0.15, color='#475569')
    
    plt.tight_layout(pad=2.0)
    return fig


def plot_energy_distribution(features, figsize=(6, 4)):
    """
    Plot the frequency band energy distribution as a donut chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0F0F1A')
    ax.set_facecolor('#0F0F1A')
    
    sizes = [
        features['low_frequency_ratio'],
        features['mid_frequency_ratio'],
        features['high_frequency_ratio']
    ]
    labels = ['Low Freq', 'Mid Freq', 'High Freq']
    colors = ['#06B6D4', '#8B5CF6', '#F43F5E']
    explode = (0.02, 0.02, 0.02)
    
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': '#E2E8F0', 'fontsize': 10},
        pctdistance=0.75
    )
    
    for text in autotexts:
        text.set_fontweight('bold')
    
    # Draw center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.50, fc='#0F0F1A')
    ax.add_artist(centre_circle)
    
    ax.set_title('Frequency Band Distribution', 
                 color='#E2E8F0', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig
