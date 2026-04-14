"""
📚 How It Works Page
Educational page explaining the technology behind AI image detection.
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="How It Works - AI vs Real",
    page_icon="📚",
    layout="wide",
)

from app import inject_custom_css, render_sidebar
inject_custom_css()
render_sidebar()

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="hero-title" style="font-size: 2.4rem;">📚 How It Works</div>
<div class="hero-subtitle">Understanding the technology behind AI image detection</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# Section 1: The Problem
# ============================================================
st.markdown('<div class="section-header">🎯 The Problem</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p style="color: #E2E8F0; font-size: 1.05rem; line-height: 1.8;">
        AI image generators like <strong style="color: #8B5CF6;">DALL-E</strong>, 
        <strong style="color: #06B6D4;">Midjourney</strong>, 
        <strong style="color: #10B981;">Stable Diffusion</strong>, and 
        <strong style="color: #F59E0B;">Google Gemini</strong> have become incredibly sophisticated. 
        They can create photorealistic images that are nearly indistinguishable from real photographs 
        to the human eye.
    </p>
    <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.7; margin-top: 1rem;">
        This poses challenges for media integrity, journalism, academic research, legal evidence, 
        and social media authenticity. Our detector addresses this by combining multiple analysis 
        techniques to identify AI-generated images.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Section 2: Our Approach
# ============================================================
st.markdown('<div class="section-header">🔬 Our Three-Layer Detection Approach</div>', 
           unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="glass-card" style="min-height: 400px;">
        <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">🧠</div>
        <h3 style="color: #8B5CF6; text-align: center;">Layer 1: Deep Learning</h3>
        <p style="color: #94A3B8; line-height: 1.7;">
            <strong style="color: #E2E8F0;">EfficientNetV2-B0</strong> is a state-of-the-art 
            convolutional neural network pre-trained on ImageNet (14M images).
        </p>
        <hr style="border-color: rgba(139,92,246,0.2);">
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            <strong style="color: #E2E8F0;">How it works:</strong><br>
            1. The base model learns visual features (edges, textures, patterns)<br>
            2. We freeze these layers and add a custom classification head<br>
            3. Train the head on 120K real vs AI images<br>
            4. Fine-tune the top layers for maximum accuracy<br>
            5. The model learns to detect subtle differences invisible to humans
        </p>
        <hr style="border-color: rgba(139,92,246,0.2);">
        <p style="color: #10B981; font-size: 0.9rem;">
            ✅ 95-97% accuracy on test set
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card" style="min-height: 400px;">
        <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">📡</div>
        <h3 style="color: #06B6D4; text-align: center;">Layer 2: Frequency Analysis</h3>
        <p style="color: #94A3B8; line-height: 1.7;">
            <strong style="color: #E2E8F0;">Fast Fourier Transform (FFT)</strong> converts 
            images from the spatial domain to the frequency domain.
        </p>
        <hr style="border-color: rgba(6,182,212,0.2);">
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            <strong style="color: #E2E8F0;">Why it works:</strong><br>
            • AI generators produce images through iterative denoising processes<br>
            • This creates subtle periodic patterns in the frequency spectrum<br>
            • Real photos have natural, continuous frequency distributions<br>
            • AI images show unusual energy concentrations or grid patterns<br>
            • The spectral slope and entropy reveal generation artifacts
        </p>
        <hr style="border-color: rgba(6,182,212,0.2);">
        <p style="color: #F59E0B; font-size: 0.9rem;">
            ⚠️ Supplementary signal — best used with other layers
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="glass-card" style="min-height: 400px;">
        <div style="text-align: center; font-size: 3rem; margin-bottom: 1rem;">📋</div>
        <h3 style="color: #10B981; text-align: center;">Layer 3: Metadata Inspection</h3>
        <p style="color: #94A3B8; line-height: 1.7;">
            <strong style="color: #E2E8F0;">EXIF metadata</strong> is embedded in images by 
            cameras and provides a "digital fingerprint."
        </p>
        <hr style="border-color: rgba(16,185,129,0.2);">
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            <strong style="color: #E2E8F0;">What we check:</strong><br>
            • Camera make and model information<br>
            • Exposure settings (aperture, shutter speed, ISO)<br>
            • GPS location data<br>
            • Date and time of capture<br>
            • Software used (e.g., "Stable Diffusion")<br>
            • Lens information and flash settings
        </p>
        <hr style="border-color: rgba(16,185,129,0.2);">
        <p style="color: #10B981; font-size: 0.9rem;">
            ✅ Real photos usually have camera EXIF; AI images typically don't
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Section 3: Transfer Learning
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🔄 Transfer Learning Explained</div>', 
           unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p style="color: #E2E8F0; font-size: 1rem; line-height: 1.8;">
        <strong style="color: #8B5CF6;">Transfer Learning</strong> is the key technique that 
        makes this project possible with limited data. Instead of training a neural network 
        from scratch (which requires millions of images and massive compute), we start with 
        a model that has already learned to understand visual features.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #8B5CF6;">Phase 1: Frozen Base Training</h4>
        <p style="color: #94A3B8; line-height: 1.6; font-size: 0.95rem;">
            • Load EfficientNetV2-B0 with ImageNet weights<br>
            • <strong style="color: #E2E8F0;">Freeze all base layers</strong> — keep learned features<br>
            • Add custom layers: GlobalAvgPool → Dense(256) → Dropout → Dense(1)<br>
            • Train only the new layers for 5 epochs<br>
            • Learning rate: 1e-3 (relatively high for fast convergence)
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #06B6D4;">Phase 2: Fine-Tuning</h4>
        <p style="color: #94A3B8; line-height: 1.6; font-size: 0.95rem;">
            • <strong style="color: #E2E8F0;">Unfreeze top 100+ layers</strong> of the base model<br>
            • Continue training with a very low learning rate (1e-5)<br>
            • This allows the model to adapt high-level features to our task<br>
            • Early stopping prevents overfitting<br>
            • Train for up to 15 more epochs
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Section 4: Grad-CAM
# ============================================================
st.markdown('<div class="section-header">🔥 Grad-CAM Explainability</div>', 
           unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <p style="color: #E2E8F0; font-size: 1rem; line-height: 1.8;">
        <strong style="color: #F59E0B;">Gradient-weighted Class Activation Mapping (Grad-CAM)</strong> 
        is a technique that makes our model's decisions interpretable. It generates a heatmap 
        showing which regions of the image the model focused on to make its prediction.
    </p>
    <br>
    <p style="color: #94A3B8; line-height: 1.7;">
        <strong style="color: #E2E8F0;">How Grad-CAM works:</strong><br>
        1. Forward pass the image through the model<br>
        2. Compute gradients of the output class w.r.t. the last convolutional layer<br>
        3. Global average pool the gradients to get importance weights<br>
        4. Weight the feature maps by these importance weights<br>
        5. Apply ReLU to get only positive contributions<br>
        6. Overlay the heatmap on the original image
    </p>
    <br>
    <p style="color: #94A3B8; font-size: 0.9rem;">
        <strong>Warm colors (red/yellow)</strong> = Areas the model found most suspicious or important<br>
        <strong>Cool colors (blue/green)</strong> = Areas with less influence on the decision
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Section 5: Dataset
# ============================================================
st.markdown('<div class="section-header">📦 Training Dataset: CIFAKE</div>', 
           unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #8B5CF6;">Dataset Overview</h4>
        <table style="width: 100%; color: #E2E8F0; border-collapse: collapse; font-size: 0.95rem;">
            <tr style="border-bottom: 1px solid rgba(139,92,246,0.15);">
                <td style="padding: 0.6rem; color: #94A3B8;">Total Images</td>
                <td style="padding: 0.6rem; font-weight: 600;">120,000</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139,92,246,0.15);">
                <td style="padding: 0.6rem; color: #94A3B8;">Real Images</td>
                <td style="padding: 0.6rem; font-weight: 600;">60,000 (from CIFAR-10)</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139,92,246,0.15);">
                <td style="padding: 0.6rem; color: #94A3B8;">AI-Generated</td>
                <td style="padding: 0.6rem; font-weight: 600;">60,000 (Stable Diffusion v1.4)</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139,92,246,0.15);">
                <td style="padding: 0.6rem; color: #94A3B8;">Original Size</td>
                <td style="padding: 0.6rem; font-weight: 600;">32×32 RGB</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139,92,246,0.15);">
                <td style="padding: 0.6rem; color: #94A3B8;">Training Split</td>
                <td style="padding: 0.6rem; font-weight: 600;">100,000 images</td>
            </tr>
            <tr>
                <td style="padding: 0.6rem; color: #94A3B8;">Test Split</td>
                <td style="padding: 0.6rem; font-weight: 600;">20,000 images</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #06B6D4;">10 Object Categories</h4>
        <p style="color: #94A3B8; line-height: 2;">
            ✈️ Airplane &nbsp;&nbsp; 🚗 Automobile &nbsp;&nbsp; 🐦 Bird<br>
            🐱 Cat &nbsp;&nbsp; 🦌 Deer &nbsp;&nbsp; 🐕 Dog<br>
            🐸 Frog &nbsp;&nbsp; 🐴 Horse &nbsp;&nbsp; 🚢 Ship<br>
            🚛 Truck
        </p>
        <hr style="border-color: rgba(6,182,212,0.2);">
        <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
            <strong style="color: #E2E8F0;">Reference:</strong><br>
            Bird, J.J. and Lotfi, A., 2024. "CIFAKE: Image Classification 
            and Explainable Identification of AI-Generated Synthetic Images." 
            <em>IEEE Access</em>.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# Section 6: Limitations
# ============================================================
st.markdown('<div class="section-header">⚠️ Limitations & Challenges</div>', 
           unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div>
            <h4 style="color: #F43F5E;">Known Limitations</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.8;">
                ❌ May struggle with images from very new AI generators<br>
                ❌ Heavily compressed images lose detection signals<br>
                ❌ Screenshots of AI images may be harder to detect<br>
                ❌ Real photos edited with AI tools create ambiguity<br>
                ❌ Low-resolution images have fewer detectable features
            </p>
        </div>
        <div>
            <h4 style="color: #10B981;">Strengths</h4>
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.8;">
                ✅ Multi-layer approach reduces false positives<br>
                ✅ Grad-CAM provides visual explainability<br>
                ✅ Works across multiple AI generator types<br>
                ✅ Metadata analysis catches obvious AI signatures<br>
                ✅ Frequency analysis detects hidden artifacts
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.info("💡 **Important:** No AI detection tool is 100% accurate. This system provides a "
        "probabilistic assessment and should be one input among many when evaluating image authenticity.")
