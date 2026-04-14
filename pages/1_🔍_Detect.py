"""
🔍 Detect Page
Main detection interface for AI vs Real image classification.
"""

import streamlit as st
import numpy as np
import os
import sys
import time
import json
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.image_preprocessing import (
    validate_image, load_image, preprocess_for_model,
    preprocess_for_display, preprocess_for_frequency, get_image_info
)
from utils.frequency_analysis import (
    compute_spectral_features, plot_frequency_spectrum, plot_energy_distribution
)
from utils.metadata_inspector import analyze_metadata, format_exif_for_display
from utils.gradcam import generate_dummy_heatmap, overlay_gradcam, create_gradcam_figure

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Detect - AI vs Real",
    page_icon="🔍",
    layout="wide",
)

# Inject CSS from main app
from app import inject_custom_css, render_sidebar

inject_custom_css()
settings = render_sidebar()


# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained EfficientNetV2 model."""
    model_path = os.path.join('models', 'saved_model', 'ai_vs_real_efficientnet.keras')
    
    if os.path.exists(model_path):
        try:
            import keras
            model = keras.saving.load_model(model_path)
            return model, True
        except Exception:
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                return model, True
            except Exception as e:
                st.warning(f"Error loading model: {e}")
                return None, False
    return None, False


def predict_image(model, img_array):
    """Run prediction on preprocessed image. Returns raw model output."""
    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    return confidence


def classify_with_ensemble(model_confidence, freq_features, meta_result, has_model):
    """
    Combine all three analysis layers into a final robust prediction.
    
    The CIFAKE-trained model works well on CIFAKE-style images but is biased
    on real-world high-resolution images (tends to output low values ~0.2 for
    everything). Therefore we use a smart ensemble:
    
    1. Metadata is the STRONGEST signal (EXIF data = phone photo = Real)
    2. Frequency analysis provides supplementary signal
    3. Model provides supplementary signal (most useful for CIFAKE-style images)
    
    Smart overrides:
    - Phone photo with 5+ EXIF camera tags → always Real
    - AI software signature in EXIF → always AI-Generated
    - No EXIF at all → lean toward AI-Generated
    """
    
    # --- Layer 1: Model Score ---
    if has_model and model_confidence is not None:
        model_raw = model_confidence
    else:
        model_raw = None
    
    # --- Layer 2: Frequency Score ---
    freq_anomaly = freq_features['anomaly_score']
    freq_real_score = 1.0 - freq_anomaly  
    
    # --- Layer 3: Metadata Score ---
    meta_real_score = meta_result['metadata_score']
    has_exif = meta_result['has_exif']
    camera_count = meta_result['camera_indicator_count']
    ai_sigs = meta_result['ai_signatures']
    
    # ===================================
    # Smart Override Rules
    # ===================================
    
    # Rule 1: AI software signature detected → definitely AI
    if ai_sigs:
        label = "AI-Generated"
        confidence = 0.95
        details = _build_details(model_raw, freq_real_score, meta_real_score, 
                                 0.05, has_model, "AI signature override")
        return label, confidence, details
    
    # Rule 2: Strong camera EXIF (5+ camera tags) → definitely Real
    if camera_count >= 5:
        label = "Real"
        confidence = max(0.85, meta_real_score)
        details = _build_details(model_raw, freq_real_score, meta_real_score,
                                 confidence, has_model, "Strong EXIF override")
        return label, confidence, details
    
    # Rule 3: Has camera Make+Model → likely Real
    if 'Make' in meta_result.get('exif_data', {}) and 'Model' in meta_result.get('exif_data', {}):
        label = "Real"
        confidence = max(0.78, meta_real_score)
        details = _build_details(model_raw, freq_real_score, meta_real_score,
                                 confidence, has_model, "Camera Make/Model override")
        return label, confidence, details
    
    # ===================================
    # Weighted Ensemble (no strong override)
    # ===================================
    
    if has_model and model_raw is not None:
        # Model is biased on out-of-distribution images (CIFAKE = 32x32)
        # Frequency analysis has uncalibrated thresholds
        # Metadata + image properties is the most reliable signal
        weights = {
            'model': 0.15,
            'frequency': 0.10,
            'metadata': 0.75,
        }
        final_real_score = (
            weights['model'] * model_raw +
            weights['frequency'] * freq_real_score +
            weights['metadata'] * meta_real_score
        )
    else:
        weights = {'frequency': 0.20, 'metadata': 0.80}
        final_real_score = (
            weights['frequency'] * freq_real_score +
            weights['metadata'] * meta_real_score
        )
    
    # --- No EXIF but image properties already factored into meta_real_score ---
    # Light penalty only (image property analysis already adjusts the score)
    if not has_exif:
        final_real_score *= 0.90
    
    # --- Final Decision ---
    if final_real_score > 0.5:
        label = "Real"
        confidence = final_real_score
    else:
        label = "AI-Generated"
        confidence = 1.0 - final_real_score
    
    details = _build_details(model_raw, freq_real_score, meta_real_score,
                             final_real_score, has_model, "Weighted ensemble")
    
    return label, confidence, details


def _build_details(model_raw, freq_real, meta_real, final_real, has_model, method):
    """Helper to build the details dict for debug display."""
    return {
        'model_raw': model_raw if has_model else None,
        'model_real_score': model_raw,
        'freq_real_score': freq_real,
        'meta_real_score': meta_real,
        'final_real_score': final_real,
        'method': method,
    }



# ============================================================
# Visualization Components
# ============================================================
def create_confidence_gauge(label, confidence):
    """Create an animated confidence gauge using Plotly."""
    color = "#10B981" if label == "Real" else "#EF4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={
            'suffix': '%',
            'font': {'size': 48, 'color': color, 'family': 'Inter'}
        },
        title={
            'text': f"<b>{label}</b>",
            'font': {'size': 24, 'color': color, 'family': 'Inter'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "#334155",
                'tickfont': {'color': '#94A3B8', 'size': 11}
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(26, 26, 46, 0.5)",
            'borderwidth': 1,
            'bordercolor': "rgba(139, 92, 246, 0.2)",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(239, 68, 68, 0.08)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.08)'},
                {'range': [66, 100], 'color': 'rgba(16, 185, 129, 0.08)'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        font={'family': 'Inter'}
    )
    
    return fig


def create_analysis_breakdown(label, model_conf, freq_features, meta_result, has_model):
    """Create a breakdown chart of all analysis components."""
    categories = []
    scores = []
    colors = []
    
    if has_model:
        categories.append("Deep Learning Model")
        scores.append(model_conf * 100 if label == "Real" else (1 - model_conf) * 100)
        colors.append("#8B5CF6")
    
    categories.append("Frequency Analysis")
    freq_score = (1 - freq_features['anomaly_score']) if label == "Real" else freq_features['anomaly_score']
    scores.append(freq_score * 100)
    colors.append("#06B6D4")
    
    categories.append("Metadata Analysis")
    meta_score = meta_result['metadata_score'] if label == "Real" else (1 - meta_result['metadata_score'])
    scores.append(meta_score * 100)
    colors.append("#10B981")
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker_color=colors,
        marker_line_color=colors,
        marker_line_width=1,
        text=[f'{s:.1f}%' for s in scores],
        textposition='outside',
        textfont=dict(color='#E2E8F0', size=13, family='Inter'),
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=10, r=60, t=30, b=10),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor='rgba(139, 92, 246, 0.08)',
            tickfont=dict(color='#94A3B8', size=10),
            title=dict(text='Confidence (%)', font=dict(color='#94A3B8', size=11))
        ),
        yaxis=dict(
            tickfont=dict(color='#E2E8F0', size=12, family='Inter'),
        ),
        title=dict(
            text='Analysis Breakdown',
            font=dict(color='#E2E8F0', size=15, family='Inter'),
            x=0.5
        )
    )
    
    return fig


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="hero-title" style="font-size: 2.4rem;">🔍 Image Detection</div>
<div class="hero-subtitle">Upload an image to determine if it's AI-generated or a real photograph</div>
""", unsafe_allow_html=True)

# Model status
model, model_loaded = load_model()

if model_loaded:
    st.success("✅ **Model loaded successfully!** Full deep learning analysis available.")
else:
    st.info("⚡ **Demo Mode** — The trained model is not yet loaded. Analysis uses frequency + metadata methods. "
            "Train the model using the notebook in `models/` for full accuracy.")

st.markdown("---")

# ============================================================
# File Upload
# ============================================================
uploaded_file = st.file_uploader(
    "📤 Upload an image for analysis",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supports JPG, PNG, WEBP, BMP formats. Max 50MB.",
    key="image_uploader"
)

if uploaded_file is not None:
    # Validate
    is_valid, error_msg = validate_image(uploaded_file)
    
    if not is_valid:
        st.error(f"❌ {error_msg}")
    else:
        # Load image
        img = load_image(uploaded_file)
        img_display = preprocess_for_display(img)
        
        # Show progress
        progress_bar = st.progress(0, text="🔄 Analyzing image...")
        
        # ==========================================
        # Analysis Pipeline (3-Layer Ensemble)
        # ==========================================
        
        # Step 1: Deep Learning Model
        progress_bar.progress(15, text="🧠 Running deep learning model...")
        time.sleep(0.3)
        
        model_conf = None
        if model_loaded:
            img_preprocessed = preprocess_for_model(img)
            model_conf = predict_image(model, img_preprocessed)
        
        # Step 2: Frequency analysis
        progress_bar.progress(35, text="📡 Running frequency analysis...")
        time.sleep(0.3)
        
        img_gray = preprocess_for_frequency(img)
        freq_features = compute_spectral_features(img_gray)
        
        # Step 3: Metadata inspection
        progress_bar.progress(55, text="📋 Inspecting metadata...")
        time.sleep(0.3)
        
        uploaded_file.seek(0)
        from PIL import Image as PILImage
        img_for_exif = PILImage.open(uploaded_file)
        meta_result = analyze_metadata(img_for_exif)
        
        # Step 4: Ensemble Classification (combine all 3 layers)
        progress_bar.progress(75, text="🔗 Combining analysis layers...")
        time.sleep(0.3)
        
        label, confidence, ensemble_details = classify_with_ensemble(
            model_conf, freq_features, meta_result, model_loaded
        )
        
        # Step 5: Grad-CAM
        progress_bar.progress(90, text="🔥 Generating attention heatmap...")
        time.sleep(0.3)
        
        if model_loaded and settings.get('run_gradcam', True):
            try:
                from utils.gradcam import get_gradcam_heatmap
                img_preprocessed = preprocess_for_model(img)
                heatmap = get_gradcam_heatmap(model, img_preprocessed)
            except Exception:
                heatmap = generate_dummy_heatmap()
        else:
            heatmap = generate_dummy_heatmap()
        
        progress_bar.progress(100, text="✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        
        # Show debug info in expander
        with st.expander("🔧 Debug: Component Scores", expanded=False):
            dbg_cols = st.columns(3)
            with dbg_cols[0]:
                if model_conf is not None:
                    st.metric("Model Raw Output", f"{model_conf:.4f}")
                    st.caption("Close to 1 = Real, Close to 0 = AI")
                else:
                    st.metric("Model", "N/A")
            with dbg_cols[1]:
                st.metric("Freq Real Score", f"{ensemble_details['freq_real_score']:.4f}")
                st.caption(f"Anomaly: {freq_features['anomaly_score']:.4f}")
            with dbg_cols[2]:
                st.metric("Meta Real Score", f"{ensemble_details['meta_real_score']:.4f}")
                st.caption(f"EXIF: {'Yes' if meta_result['has_exif'] else 'No'}")
        
        # ==========================================
        # Results Display
        # ==========================================
        
        # Main Result Card
        result_class = "result-real" if label == "Real" else "result-ai"
        emoji = "📷" if label == "Real" else "🤖"
        
        st.markdown(f"""
        <div class="{result_class}">
            <div class="result-label">{emoji} {label.upper()}</div>
            <div class="result-confidence">{confidence:.1%}</div>
            <p style="color: #94A3B8; margin-top: 0.5rem;">
                {'This image appears to be a genuine photograph captured by a camera.' 
                 if label == 'Real' else 
                 'This image appears to be generated by an AI model.'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Two-column layout: Image + Gauge
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="section-header">📸 Uploaded Image</div>', 
                       unsafe_allow_html=True)
            st.image(img_display, use_container_width=True)
            
            # Image info
            info = get_image_info(img)
            info_cols = st.columns(3)
            for i, (key, val) in enumerate(list(info.items())[:3]):
                with info_cols[i]:
                    st.metric(key, val)
        
        with col2:
            st.markdown('<div class="section-header">📊 Confidence Score</div>', 
                       unsafe_allow_html=True)
            gauge_fig = create_confidence_gauge(label, confidence)
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Analysis breakdown
            breakdown_fig = create_analysis_breakdown(
                label, confidence, freq_features, meta_result, model_loaded
            )
            st.plotly_chart(breakdown_fig, use_container_width=True)
        
        st.markdown("---")
        
        # ==========================================
        # Detailed Analysis Tabs
        # ==========================================
        st.markdown('<div class="section-header">🔬 Detailed Analysis</div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📡 Frequency Analysis",
            "📋 Metadata",
            "🔥 Grad-CAM",
            "📊 Summary"
        ])
        
        # Tab 1: Frequency Analysis
        with tab1:
            if settings.get('run_frequency', True):
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #06B6D4;">Frequency Domain Analysis (FFT)</h4>
                    <p style="color: #94A3B8; font-size: 0.9rem;">
                        AI-generated images often exhibit distinct patterns in the frequency domain. 
                        This analysis applies a 2D Fast Fourier Transform to detect spectral anomalies.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Spectrum plots
                fig_spectrum = plot_frequency_spectrum(img_gray)
                st.pyplot(fig_spectrum)
                
                # Energy distribution
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    fig_energy = plot_energy_distribution(freq_features)
                    st.pyplot(fig_energy)
                
                with col_b:
                    st.markdown("""
                    <div class="glass-card">
                        <h4 style="color: #E2E8F0;">Spectral Features</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    feat_cols = st.columns(2)
                    with feat_cols[0]:
                        st.metric("Low Freq Ratio", f"{freq_features['low_frequency_ratio']:.3f}")
                        st.metric("Mid Freq Ratio", f"{freq_features['mid_frequency_ratio']:.3f}")
                        st.metric("High Freq Ratio", f"{freq_features['high_frequency_ratio']:.3f}")
                    with feat_cols[1]:
                        st.metric("Spectral Entropy", f"{freq_features['spectral_entropy']:.2f}")
                        st.metric("Spectral Slope", f"{freq_features['spectral_slope']:.3f}")
                        st.metric("Anomaly Score", f"{freq_features['anomaly_score']:.3f}")
                    
                    if freq_features['anomaly_indicators']:
                        st.warning("**Anomaly Indicators:**\n" + 
                                   "\n".join(f"- {ind}" for ind in freq_features['anomaly_indicators']))
            else:
                st.info("Frequency analysis is disabled. Enable it in the sidebar.")
        
        # Tab 2: Metadata
        with tab2:
            if settings.get('run_metadata', True):
                st.markdown(f"""
                <div class="glass-card">
                    <h4 style="color: #10B981;">Metadata Verdict</h4>
                    <p style="color: #E2E8F0; font-size: 1rem;">{meta_result['verdict']}</p>
                    <p style="color: #94A3B8;">Score: <strong style="color: #8B5CF6;">{meta_result['metadata_score']:.2f}</strong> / 1.00 
                    (Higher = More likely Real)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Details
                if meta_result['details']:
                    st.markdown("**Analysis Points:**")
                    for detail in meta_result['details']:
                        st.markdown(f"  {detail}")
                
                # EXIF table
                if meta_result['has_exif']:
                    st.markdown("---")
                    st.markdown("**📋 EXIF Data:**")
                    exif_display = format_exif_for_display(meta_result['exif_data'])
                    if exif_display:
                        import pandas as pd
                        df = pd.DataFrame(exif_display, columns=['Tag', 'Value'])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No readable EXIF tags found.")
                else:
                    st.warning("⚠️ No EXIF metadata found in this image. "
                              "AI-generated images typically lack camera metadata.")
            else:
                st.info("Metadata inspection is disabled. Enable it in the sidebar.")
        
        # Tab 3: Grad-CAM
        with tab3:
            if settings.get('run_gradcam', True):
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #F59E0B;">Grad-CAM Attention Map</h4>
                    <p style="color: #94A3B8; font-size: 0.9rem;">
                        Gradient-weighted Class Activation Mapping shows which regions of the image 
                        the model focused on to make its prediction. Red/warm areas indicate high 
                        attention regions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if not model_loaded:
                    st.caption("*Using simulated heatmap in demo mode. Train the model for actual Grad-CAM.*")
                
                # Display Grad-CAM
                gradcam_fig = create_gradcam_figure(img_display, heatmap, label, confidence)
                st.pyplot(gradcam_fig)
                
                # Overlay with slider
                st.markdown("**Adjust Heatmap Overlay Intensity:**")
                alpha = st.slider("Overlay Alpha", 0.0, 1.0, 0.4, 0.05, key="alpha_slider")
                overlaid_img = overlay_gradcam(img_display, heatmap, alpha=alpha)
                st.image(overlaid_img, caption=f"Heatmap Overlay (α={alpha})", 
                        use_container_width=True)
            else:
                st.info("Grad-CAM is disabled. Enable it in the sidebar.")
        
        # Tab 4: Summary
        with tab4:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #8B5CF6;">Complete Analysis Summary</h4>
            </div>
            """, unsafe_allow_html=True)
            
            summary_data = {
                "Category": [
                    "Final Verdict",
                    "Confidence",
                    "Model Used",
                    "Frequency Anomaly Score",
                    "Metadata Score",
                    "EXIF Data Present",
                    "Camera Indicators",
                    "AI Software Signatures",
                    "Image Dimensions",
                    "File Format"
                ],
                "Result": [
                    f"{'📷 Real Photo' if label == 'Real' else '🤖 AI-Generated'}",
                    f"{confidence:.1%}",
                    "EfficientNetV2-B0" if model_loaded else "Demo (Freq + Meta)",
                    f"{freq_features['anomaly_score']:.3f}",
                    f"{meta_result['metadata_score']:.2f}",
                    "✅ Yes" if meta_result['has_exif'] else "❌ No",
                    str(meta_result['camera_indicator_count']),
                    ", ".join(meta_result['ai_signatures']) if meta_result['ai_signatures'] else "None",
                    f"{img.size[0]} × {img.size[1]}",
                    getattr(img, 'format', 'Unknown') or 'Unknown'
                ]
            }
            
            import pandas as pd
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download results
            st.markdown("---")
            results_json = json.dumps({
                "verdict": label,
                "confidence": confidence,
                "frequency_analysis": {
                    "anomaly_score": freq_features['anomaly_score'],
                    "spectral_entropy": freq_features['spectral_entropy'],
                    "spectral_slope": freq_features['spectral_slope'],
                    "low_freq": freq_features['low_frequency_ratio'],
                    "mid_freq": freq_features['mid_frequency_ratio'],
                    "high_freq": freq_features['high_frequency_ratio'],
                },
                "metadata_analysis": {
                    "score": meta_result['metadata_score'],
                    "has_exif": meta_result['has_exif'],
                    "camera_indicators": meta_result['camera_indicator_count'],
                    "verdict": meta_result['verdict'],
                }
            }, indent=2)
            
            st.download_button(
                "📥 Download Analysis Report (JSON)",
                results_json,
                file_name="analysis_report.json",
                mime="application/json"
            )

else:
    # Show helpful message when no image is uploaded
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <h3 style="color: #E2E8F0; margin-bottom: 0.5rem;">Drop an image here to analyze</h3>
        <p style="color: #94A3B8;">
            Upload any image — AI-generated from ChatGPT, Gemini, Midjourney, DALL-E, 
            Stable Diffusion, or a real photo from your camera.
        </p>
        <p style="color: #64748B; font-size: 0.85rem; margin-top: 1rem;">
            Supported formats: JPG, PNG, WEBP, BMP • Max size: 50MB
        </p>
    </div>
    """, unsafe_allow_html=True)
