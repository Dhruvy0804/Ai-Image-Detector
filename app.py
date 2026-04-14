"""
AI vs Real Image Detector - Main Streamlit Application
======================================================

A deep learning-powered web app that detects whether an uploaded
image is AI-generated or a real photograph.

Features:
- EfficientNetV2-B0 classification
- Frequency domain analysis (FFT)
- EXIF metadata inspection
- Grad-CAM explainability
- Beautiful dark-mode UI

Usage:
    streamlit run app.py
"""

import streamlit as st
import os

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS — Premium Dark Theme
# ============================================================
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== Global Styles ===== */
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
    }
    
    /* ===== Sidebar Styling ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #8B5CF6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* ===== Header / Hero ===== */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8B5CF6 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* ===== Glass Cards ===== */
    .glass-card {
        background: rgba(26, 26, 46, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
        transform: translateY(-2px);
    }
    
    /* ===== Result Cards ===== */
    .result-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    
    .result-ai {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(244, 63, 94, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    .result-confidence {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .result-real .result-label { color: #10B981; }
    .result-real .result-confidence { color: #34D399; }
    .result-ai .result-label { color: #EF4444; }
    .result-ai .result-confidence { color: #F87171; }
    
    /* ===== Metrics Grid ===== */
    .metric-card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #8B5CF6;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ===== Section Headers ===== */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E2E8F0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(139, 92, 246, 0.2);
    }
    
    /* ===== Feature Cards (Home) ===== */
    .feature-card {
        background: rgba(26, 26, 46, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.15);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* ===== Status Badge ===== */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-demo {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* ===== File Uploader ===== */
    .stFileUploader > div {
        border: 2px dashed rgba(139, 92, 246, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(26, 26, 46, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(139, 92, 246, 0.6) !important;
        background: rgba(26, 26, 46, 0.6) !important;
    }
    
    /* ===== Progress Bar ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8B5CF6, #06B6D4) !important;
    }
    
    /* ===== Tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background: rgba(26, 26, 46, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.1);
        color: #94A3B8;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(139, 92, 246, 0.15) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
        color: #E2E8F0 !important;
    }
    
    /* ===== Info/Alert Boxes ===== */
    .stAlert {
        border-radius: 12px !important;
    }
    
    /* ===== Divider ===== */
    hr {
        border: none;
        border-top: 1px solid rgba(139, 92, 246, 0.15);
        margin: 1.5rem 0;
    }
    
    /* ===== Buttons ===== */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(6, 182, 212, 0.1));
        color: #E2E8F0;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 0.5rem 1.5rem;
    }
    
    .stButton > button:hover {
        border-color: rgba(139, 92, 246, 0.6);
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(6, 182, 212, 0.15));
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.2);
    }
    
    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F0F1A;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.5);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("# 🔍 AI Detector")
        st.markdown("---")
        
        # Model status
        model_path = os.path.join('models', 'saved_model', 'ai_vs_real_efficientnet.keras')
        if os.path.exists(model_path):
            st.markdown('<span class="status-badge badge-ready">✅ Model Loaded</span>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge badge-demo">⚡ Demo Mode</span>', 
                       unsafe_allow_html=True)
            st.caption("Train the model to unlock full accuracy. The app works in demo mode with frequency & metadata analysis.")
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Analysis Settings")
        
        run_frequency = st.toggle("Frequency Analysis", value=True, 
                                   help="Run FFT-based spectral analysis")
        run_metadata = st.toggle("Metadata Inspection", value=True,
                                  help="Analyze EXIF metadata")
        run_gradcam = st.toggle("Grad-CAM Heatmap", value=True,
                                 help="Show model attention visualization")
        
        st.markdown("---")
        
        st.markdown("### 📊 Quick Stats")
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">95%+</div>
            <div class="metric-label">Target Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3-Layer</div>
            <div class="metric-label">Analysis Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(
            "<p style='text-align: center; color: #64748B; font-size: 0.75rem;'>"
            "Built with EfficientNetV2 + Streamlit<br>"
            "© 2026 AI vs Real Detector</p>",
            unsafe_allow_html=True
        )
        
        return {
            'run_frequency': run_frequency,
            'run_metadata': run_metadata,
            'run_gradcam': run_gradcam,
        }


# ============================================================
# Main Home Page
# ============================================================
def main():
    inject_custom_css()
    settings = render_sidebar()
    
    # Store settings in session state
    st.session_state['analysis_settings'] = settings
    
    # Hero Section
    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
        <div class="hero-title">AI vs Real Image Detector</div>
        <div class="hero-subtitle">
            Detect AI-generated images using deep learning, frequency analysis, and metadata inspection
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Deep Learning</div>
            <div class="feature-desc">EfficientNetV2 fine-tuned on 120K+ images for accurate classification</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📡</div>
            <div class="feature-title">Frequency Analysis</div>
            <div class="feature-desc">FFT-based spectral analysis to detect generation artifacts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <div class="feature-title">Metadata Inspector</div>
            <div class="feature-desc">EXIF data analysis to verify camera origin and authenticity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔥</div>
            <div class="feature-title">Explainable AI</div>
            <div class="feature-desc">Grad-CAM heatmaps showing what the model focuses on</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How to Use
    st.markdown('<div class="section-header">🚀 How to Get Started</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #8B5CF6;">Step 1</h3>
            <p style="color: #E2E8F0; font-size: 1.1rem; font-weight: 600;">Navigate to Detect</p>
            <p style="color: #94A3B8;">Go to the <strong>🔍 Detect</strong> page from the sidebar to start analyzing images.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #06B6D4;">Step 2</h3>
            <p style="color: #E2E8F0; font-size: 1.1rem; font-weight: 600;">Upload an Image</p>
            <p style="color: #94A3B8;">Upload any image — a photo from your camera or an AI-generated image from anywhere.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #10B981;">Step 3</h3>
            <p style="color: #E2E8F0; font-size: 1.1rem; font-weight: 600;">Get Results</p>
            <p style="color: #94A3B8;">View the AI vs Real verdict with confidence scores, heatmaps, and detailed analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tech Stack
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚙️ Technology Stack</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <table style="width: 100%; color: #E2E8F0; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid rgba(139, 92, 246, 0.15);">
                <td style="padding: 0.8rem; font-weight: 600; color: #8B5CF6;">🧠 Model</td>
                <td style="padding: 0.8rem;">EfficientNetV2-B0 (Transfer Learning from ImageNet)</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139, 92, 246, 0.15);">
                <td style="padding: 0.8rem; font-weight: 600; color: #06B6D4;">📊 Framework</td>
                <td style="padding: 0.8rem;">TensorFlow / Keras</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139, 92, 246, 0.15);">
                <td style="padding: 0.8rem; font-weight: 600; color: #10B981;">🖥️ Interface</td>
                <td style="padding: 0.8rem;">Streamlit with Custom Dark Theme</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(139, 92, 246, 0.15);">
                <td style="padding: 0.8rem; font-weight: 600; color: #F59E0B;">📡 Analysis</td>
                <td style="padding: 0.8rem;">FFT Frequency Analysis + EXIF Metadata Inspection</td>
            </tr>
            <tr>
                <td style="padding: 0.8rem; font-weight: 600; color: #F43F5E;">📦 Dataset</td>
                <td style="padding: 0.8rem;">CIFAKE — 120,000 Real & AI-Generated Images</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
