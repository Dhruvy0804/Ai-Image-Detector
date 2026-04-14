"""
🎭 AI-Edit Detection Page
==========================
Detects which specific regions of an image have been AI-modified
vs. original camera-captured content.

Use case: You upload a real photo to an AI tool, ask it to change
clothes/add objects/edit something — this page shows you exactly
WHICH parts were modified by the AI.
"""

import streamlit as st
import numpy as np
import os
import sys
import time
import json
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.manipulation_detector import (
    compute_ela, compute_ela_multi_quality,
    compute_noise_variance_map, compute_noise_inconsistency,
    compute_local_frequency_map, compute_edge_inconsistency,
    compute_manipulation_map, compute_direct_diff,
    create_heatmap_overlay, create_manipulation_figure,
    plot_individual_map
)

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="AI-Edit Detection - AI vs Real",
    page_icon="🎭",
    layout="wide",
)

# Inject CSS from main app
from app import inject_custom_css, render_sidebar

inject_custom_css()
settings = render_sidebar()


# ============================================================
# Additional Custom CSS for this page
# ============================================================
st.markdown("""
<style>
    .manipulation-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(244, 63, 94, 0.10));
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .manipulation-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .manipulation-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(251, 191, 36, 0.08));
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
    }
    .technique-card {
        background: rgba(26, 26, 46, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.12);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 0.8rem;
    }
    .technique-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(139, 92, 246, 0.1);
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        margin-right: 1.5rem;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Page Header
# ============================================================
st.markdown("""
<div class="hero-title" style="font-size: 2.4rem;">🎭 AI-Edit Detection</div>
<div class="hero-subtitle">
    Detect which specific regions of an image have been AI-modified vs. original
</div>
""", unsafe_allow_html=True)

# Explanation
st.markdown("""
<div class="glass-card">
    <h4 style="color: #8B5CF6;">How It Works</h4>
    <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6;">
        When you upload a real photo to an AI tool (like ChatGPT, Midjourney, etc.) and ask it to 
        change clothes, add objects, or modify something — the AI leaves <strong style="color: #E2E8F0;">forensic traces</strong> 
        in the modified regions. This tool uses <strong style="color: #06B6D4;">4 forensic techniques</strong> to detect exactly 
        <strong style="color: #EF4444;">which parts</strong> were AI-edited.
    </p>
    <div style="margin-top: 1rem;">
        <span class="legend-item"><span class="legend-dot" style="background: #EF4444;"></span> AI-Edited Region</span>
        <span class="legend-item"><span class="legend-dot" style="background: #3B82F6;"></span> Original / Real Region</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ============================================================
# Mode Selection
# ============================================================
st.markdown('<div class="section-header">📋 Analysis Mode</div>', unsafe_allow_html=True)

mode = st.radio(
    "Choose analysis mode:",
    [
        "🔍 Single Image (Auto-detect edited regions)",
        "🔄 Compare Original + Edited (Most accurate)"
    ],
    help="Single image mode uses forensic analysis. Comparison mode gives the most accurate results when you have the original photo.",
    key="analysis_mode"
)

st.markdown("---")

# ============================================================
# Upload Section
# ============================================================
if "Compare" in mode:
    # Two-image comparison mode
    st.markdown('<div class="section-header">📤 Upload Images</div>', unsafe_allow_html=True)
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("""
        <div class="technique-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">📷</div>
            <div style="color: #10B981; font-weight: 600;">Original Photo</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">The real photo before AI editing</div>
        </div>
        """, unsafe_allow_html=True)
        original_file = st.file_uploader(
            "Upload original photo",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="original_upload"
        )

    with col_upload2:
        st.markdown("""
        <div class="technique-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🤖</div>
            <div style="color: #EF4444; font-weight: 600;">AI-Edited Image</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">The image after AI modification</div>
        </div>
        """, unsafe_allow_html=True)
        edited_file = st.file_uploader(
            "Upload AI-edited image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="edited_upload"
        )

    if original_file and edited_file:
        original_img = Image.open(original_file).convert('RGB')
        edited_img = Image.open(edited_file).convert('RGB')

        # Show both images
        col_show1, col_show2 = st.columns(2)
        with col_show1:
            st.image(original_img, caption="📷 Original Photo", use_container_width=True)
        with col_show2:
            st.image(edited_img, caption="🤖 AI-Edited Image", use_container_width=True)

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Analysis Settings", expanded=False):
            sensitivity = st.slider(
                "Detection Sensitivity",
                0.0, 1.0, 0.5, 0.05,
                help="Higher = more sensitive, may produce more false positives",
                key="sensitivity_compare"
            )
            overlay_alpha = st.slider(
                "Heatmap Overlay Opacity",
                0.0, 1.0, 0.5, 0.05,
                key="alpha_compare"
            )
            diff_amplification = st.slider(
                "Difference Amplification",
                1, 20, 5,
                help="How much to amplify pixel-level differences",
                key="amplification"
            )

        # Run analysis
        if st.button("🔍 Detect AI-Edited Regions", key="run_compare", use_container_width=True):
            progress = st.progress(0, text="🔄 Starting analysis...")

            # Step 1: Direct comparison
            progress.progress(20, text="🔄 Computing direct pixel comparison...")
            time.sleep(0.3)
            diff_map, diff_color = compute_direct_diff(
                original_img, edited_img, amplification=diff_amplification
            )

            # Step 2: Forensic analysis on edited image
            progress.progress(40, text="🔬 Running forensic analysis on edited image...")
            time.sleep(0.3)
            combined_map, individual_maps, stats = compute_manipulation_map(
                edited_img, sensitivity=sensitivity
            )

            # Step 3: Merge direct diff with forensic analysis
            progress.progress(70, text="🔗 Combining direct comparison + forensic analysis...")
            time.sleep(0.3)

            # Direct diff is the most reliable when we have the original
            h, w = np.array(edited_img.convert('L')).shape
            import cv2
            if diff_map.shape != (h, w):
                diff_map = cv2.resize(diff_map, (w, h), interpolation=cv2.INTER_LINEAR)
            if combined_map.shape != (h, w):
                combined_map = cv2.resize(combined_map, (w, h), interpolation=cv2.INTER_LINEAR)

            # Weighted merge: 70% direct diff, 30% forensic
            final_map = 0.70 * diff_map + 0.30 * combined_map
            if final_map.max() > 0:
                final_map = final_map / final_map.max()

            # Compute stats
            manipulation_pct = float(np.mean(final_map > 0.5) * 100)

            progress.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress.empty()

            # ==========================================
            # Results
            # ==========================================

            # Verdict card
            if manipulation_pct > 30:
                card_class = "manipulation-high"
                verdict_emoji = "🔴"
                verdict_text = "Significant AI Modification Detected"
                verdict_desc = f"Approximately {manipulation_pct:.1f}% of the image has been AI-edited."
            elif manipulation_pct > 10:
                card_class = "manipulation-medium"
                verdict_emoji = "🟡"
                verdict_text = "Moderate AI Modification Detected"
                verdict_desc = f"Approximately {manipulation_pct:.1f}% of the image has been AI-edited."
            else:
                card_class = "manipulation-low"
                verdict_emoji = "🟢"
                verdict_text = "Minimal Modification Detected"
                verdict_desc = f"Only {manipulation_pct:.1f}% of the image appears modified."

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{verdict_emoji}</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #E2E8F0; margin-bottom: 0.3rem;">
                    {verdict_text}
                </div>
                <div style="font-size: 2.5rem; font-weight: 800; color: #8B5CF6; margin: 0.5rem 0;">
                    {manipulation_pct:.1f}%
                </div>
                <div style="color: #94A3B8; font-size: 0.95rem;">{verdict_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Main visualization
            st.markdown('<div class="section-header">🗺️ Manipulation Heatmap</div>',
                       unsafe_allow_html=True)

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("**AI-Edited Image:**")
                st.image(edited_img, use_container_width=True)

            with col_r2:
                st.markdown("**AI-Edited Regions Highlighted:**")
                overlay = create_heatmap_overlay(edited_img, final_map, alpha=overlay_alpha)
                st.image(overlay, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Direct difference
            st.markdown('<div class="section-header">🔍 Direct Pixel Difference</div>',
                       unsafe_allow_html=True)

            col_d1, col_d2, col_d3 = st.columns(3)

            with col_d1:
                st.markdown("**Original:**")
                st.image(original_img, use_container_width=True)

            with col_d2:
                st.markdown("**AI-Edited:**")
                st.image(edited_img, use_container_width=True)

            with col_d3:
                st.markdown("**Differences (amplified):**")
                st.image(diff_color, use_container_width=True)

            st.markdown("---")

            # Detailed analysis tabs
            st.markdown('<div class="section-header">🔬 Detailed Forensic Analysis</div>',
                       unsafe_allow_html=True)

            tabs = st.tabs([
                "🔥 ELA (Error Levels)",
                "📡 Noise Analysis",
                "📊 Frequency Analysis",
                "🔗 Edge Coherence",
                "📋 Full Report"
            ])

            map_configs = [
                ('ela', '🔥 Error Level Analysis', 'hot',
                 "Shows regions with inconsistent JPEG compression error levels. "
                 "AI-edited areas typically show different error patterns than original regions."),
                ('noise', '📡 Noise Inconsistency', 'inferno',
                 "Camera sensors produce uniform noise. AI-edited regions have "
                 "different noise characteristics — either smoother or artificially textured."),
                ('frequency', '📊 Frequency Anomaly', 'magma',
                 "AI-generated textures have distinct spectral fingerprints in the frequency domain. "
                 "This analysis detects local patches with anomalous spectral characteristics."),
                ('edge', '🔗 Edge Coherence', 'plasma',
                 "Detects boundary artifacts between real and AI-edited regions. "
                 "AI edits often create subtle edge discontinuities at modification boundaries."),
            ]

            for i, (key, title, cmap, desc) in enumerate(map_configs):
                with tabs[i]:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="color: #06B6D4;">{title}</h4>
                        <p style="color: #94A3B8; font-size: 0.9rem;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if key in individual_maps:
                        fig = plot_individual_map(edited_img, individual_maps[key], title, cmap)
                        st.pyplot(fig)
                    else:
                        st.info(f"{title} was not enabled for this analysis.")

            # Full Report tab
            with tabs[4]:
                st.markdown("""
                <div class="glass-card">
                    <h4 style="color: #8B5CF6;">Complete Analysis Report</h4>
                </div>
                """, unsafe_allow_html=True)

                import pandas as pd
                report_data = {
                    "Metric": [
                        "Analysis Mode",
                        "AI-Modified Area",
                        "Max Manipulation Intensity",
                        "Mean Signal Strength",
                        "Techniques Used",
                        "Original Image Size",
                        "Edited Image Size",
                    ],
                    "Value": [
                        "Direct Comparison + Forensic",
                        f"{manipulation_pct:.1f}%",
                        f"{float(np.max(final_map)):.3f}",
                        f"{float(np.mean(final_map)):.3f}",
                        str(stats.get('num_techniques', 4)),
                        f"{original_img.size[0]} × {original_img.size[1]}",
                        f"{edited_img.size[0]} × {edited_img.size[1]}",
                    ]
                }
                st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

                # Download
                st.markdown("---")
                report_json = json.dumps({
                    "mode": "comparison",
                    "manipulation_percentage": manipulation_pct,
                    "max_intensity": float(np.max(final_map)),
                    "mean_intensity": float(np.mean(final_map)),
                    "techniques": list(individual_maps.keys()),
                    "verdict": verdict_text,
                }, indent=2)

                st.download_button(
                    "📥 Download Manipulation Report (JSON)",
                    report_json,
                    file_name="manipulation_report.json",
                    mime="application/json"
                )

else:
    # Single image mode
    st.markdown('<div class="section-header">📤 Upload AI-Edited Image</div>',
               unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload an image that may have been AI-edited",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload the image that was possibly edited by an AI tool. "
             "The tool will detect which regions were AI-modified.",
        key="single_upload"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        
        # Show image
        col_img, col_settings = st.columns([2, 1])

        with col_img:
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.caption(f"📐 Size: {img.size[0]} × {img.size[1]} px")

        with col_settings:
            st.markdown("""
            <div class="glass-card">
                <h4 style="color: #8B5CF6;">⚙️ Settings</h4>
            </div>
            """, unsafe_allow_html=True)

            sensitivity = st.slider(
                "Detection Sensitivity",
                0.0, 1.0, 0.5, 0.05,
                help="Higher = more sensitive (may flag more areas). "
                     "Lower = only the most obvious edits.",
                key="sensitivity_single"
            )

            overlay_alpha = st.slider(
                "Heatmap Overlay Opacity",
                0.0, 1.0, 0.5, 0.05,
                key="alpha_single"
            )

            st.markdown("**Enable/Disable Techniques:**")
            use_ela = st.checkbox("Error Level Analysis", value=True, key="use_ela")
            use_noise = st.checkbox("Noise Analysis", value=True, key="use_noise")
            use_freq = st.checkbox("Frequency Analysis", value=True, key="use_freq")
            use_edge = st.checkbox("Edge Coherence", value=True, key="use_edge")

        st.markdown("---")

        # Run analysis
        if st.button("🔍 Detect AI-Edited Regions", key="run_single", use_container_width=True):
            progress = st.progress(0, text="🔄 Starting forensic analysis...")

            # Run the full pipeline
            steps = []
            if use_ela:
                steps.append("ELA")
            if use_noise:
                steps.append("Noise")
            if use_freq:
                steps.append("Frequency")
            if use_edge:
                steps.append("Edge")

            total_steps = len(steps) + 1
            step_pct = 90 // max(total_steps, 1)

            current = 0
            for i, step_name in enumerate(steps):
                progress.progress(
                    min(current + step_pct, 90),
                    text=f"🔬 Running {step_name} analysis ({i+1}/{len(steps)})..."
                )
                time.sleep(0.5)
                current += step_pct

            progress.progress(50, text="🔬 Computing manipulation map...")

            combined_map, individual_maps, stats = compute_manipulation_map(
                img,
                sensitivity=sensitivity,
                use_ela=use_ela,
                use_noise=use_noise,
                use_frequency=use_freq,
                use_edge=use_edge
            )

            progress.progress(100, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress.empty()

            # ==========================================
            # Results
            # ==========================================
            manipulation_pct = stats.get('manipulation_percentage', 0)

            # Verdict card
            if manipulation_pct > 30:
                card_class = "manipulation-high"
                verdict_emoji = "🔴"
                verdict_text = "Significant AI Modification Detected"
                verdict_desc = f"Approximately {manipulation_pct:.1f}% of the image shows signs of AI editing."
            elif manipulation_pct > 10:
                card_class = "manipulation-medium"
                verdict_emoji = "🟡"
                verdict_text = "Possible AI Modification Detected"
                verdict_desc = f"About {manipulation_pct:.1f}% of the image shows potential AI editing signs."
            elif manipulation_pct > 3:
                card_class = "manipulation-medium"
                verdict_emoji = "🟡"
                verdict_text = "Minor Modifications Possible"
                verdict_desc = f"Small areas ({manipulation_pct:.1f}%) show possible editing. Could be natural variation."
            else:
                card_class = "manipulation-low"
                verdict_emoji = "🟢"
                verdict_text = "No Significant AI Editing Detected"
                verdict_desc = "The image appears largely unmodified or is entirely AI-generated/real."

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{verdict_emoji}</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #E2E8F0; margin-bottom: 0.3rem;">
                    {verdict_text}
                </div>
                <div style="font-size: 2.5rem; font-weight: 800; color: #8B5CF6; margin: 0.5rem 0;">
                    {manipulation_pct:.1f}%
                </div>
                <div style="color: #94A3B8; font-size: 0.95rem;">{verdict_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Technique contribution cards
            st.markdown('<div class="section-header">📊 Technique Contributions</div>',
                       unsafe_allow_html=True)

            tech_cols = st.columns(len(individual_maps) if individual_maps else 1)
            tech_info = {
                'ela': ('🔥', 'ELA', '#EF4444'),
                'noise': ('📡', 'Noise', '#F59E0B'),
                'frequency': ('📊', 'Frequency', '#8B5CF6'),
                'edge': ('🔗', 'Edge', '#06B6D4'),
            }

            for i, (key, m) in enumerate(individual_maps.items()):
                if i < len(tech_cols):
                    emoji, name, color = tech_info.get(key, ('🔬', key, '#8B5CF6'))
                    mean_signal = float(np.mean(m)) * 100
                    with tech_cols[i]:
                        st.markdown(f"""
                        <div class="technique-card">
                            <div style="font-size: 1.8rem;">{emoji}</div>
                            <div style="color: {color}; font-weight: 700; font-size: 1.1rem;">{name}</div>
                            <div style="color: #E2E8F0; font-size: 1.4rem; font-weight: 800; margin: 0.3rem 0;">
                                {mean_signal:.1f}%
                            </div>
                            <div style="color: #94A3B8; font-size: 0.75rem;">Average Signal</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Main heatmap
            st.markdown('<div class="section-header">🗺️ Manipulation Heatmap</div>',
                       unsafe_allow_html=True)

            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("**Original Image:**")
                st.image(img, use_container_width=True)

            with col_r2:
                st.markdown("**AI-Edited Regions Highlighted:**")
                overlay = create_heatmap_overlay(img, combined_map, alpha=overlay_alpha)
                st.image(overlay, use_container_width=True)

            # Adjustable overlay
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🎚️ Adjust Overlay Intensity:**")
            live_alpha = st.slider(
                "Overlay Alpha", 0.0, 1.0, overlay_alpha, 0.05,
                key="live_alpha"
            )
            if live_alpha != overlay_alpha:
                overlay_adjusted = create_heatmap_overlay(img, combined_map, alpha=live_alpha)
                st.image(overlay_adjusted, caption=f"Heatmap Overlay (α={live_alpha})",
                        use_container_width=True)

            st.markdown("---")

            # Detailed analysis tabs
            st.markdown('<div class="section-header">🔬 Detailed Forensic Analysis</div>',
                       unsafe_allow_html=True)

            tab_names = []
            tab_keys = []
            map_configs = [
                ('ela', '🔥 ELA', 'hot',
                 "Error Level Analysis — Detects regions with inconsistent JPEG compression. "
                 "Bright areas indicate regions that were modified after the initial compression."),
                ('noise', '📡 Noise', 'inferno',
                 "Noise Variance Map — Camera sensors produce uniform noise. AI-edited regions "
                 "have different noise characteristics. Bright areas show noise inconsistency."),
                ('frequency', '📊 Frequency', 'magma',
                 "Local Frequency Anomaly — Uses sliding-window FFT to detect patches with "
                 "different spectral characteristics. AI textures have distinct frequency signatures."),
                ('edge', '🔗 Edge', 'plasma',
                 "Edge Coherence — Detects boundary artifacts at transitions between "
                 "original and AI-edited regions. Bright areas show potential edit boundaries."),
            ]

            for key, name, _, _ in map_configs:
                if key in individual_maps:
                    tab_names.append(name)
                    tab_keys.append(key)

            tab_names.append("📋 Report")

            if tab_names:
                tabs = st.tabs(tab_names)

                for i, key in enumerate(tab_keys):
                    with tabs[i]:
                        config = next(c for c in map_configs if c[0] == key)
                        _, title, cmap, desc = config

                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="color: #06B6D4;">{title}</h4>
                            <p style="color: #94A3B8; font-size: 0.9rem;">{desc}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        fig = plot_individual_map(img, individual_maps[key], title, cmap)
                        st.pyplot(fig)

                # Report tab
                with tabs[-1]:
                    st.markdown("""
                    <div class="glass-card">
                        <h4 style="color: #8B5CF6;">Complete Analysis Report</h4>
                    </div>
                    """, unsafe_allow_html=True)

                    import pandas as pd
                    report_data = {
                        "Metric": [
                            "Analysis Mode",
                            "Verdict",
                            "AI-Modified Area",
                            "Max Manipulation Intensity",
                            "Mean Signal Strength",
                            "Techniques Used",
                            "Sensitivity",
                            "Image Size",
                        ],
                        "Value": [
                            "Single Image Forensics",
                            verdict_text,
                            f"{manipulation_pct:.1f}%",
                            f"{stats.get('max_intensity', 0):.3f}",
                            f"{stats.get('mean_intensity', 0):.3f}",
                            ", ".join(individual_maps.keys()),
                            f"{sensitivity:.2f}",
                            f"{img.size[0]} × {img.size[1]}",
                        ]
                    }
                    st.dataframe(pd.DataFrame(report_data),
                                use_container_width=True, hide_index=True)

                    # Download
                    st.markdown("---")
                    report_json = json.dumps({
                        "mode": "single_image_forensics",
                        "verdict": verdict_text,
                        "manipulation_percentage": manipulation_pct,
                        "max_intensity": stats.get('max_intensity', 0),
                        "mean_intensity": stats.get('mean_intensity', 0),
                        "sensitivity": sensitivity,
                        "techniques": list(individual_maps.keys()),
                        "image_size": list(img.size),
                    }, indent=2)

                    st.download_button(
                        "📥 Download Manipulation Report (JSON)",
                        report_json,
                        file_name="manipulation_report.json",
                        mime="application/json"
                    )

    else:
        # No image uploaded — show instructions
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎭</div>
            <h3 style="color: #E2E8F0; margin-bottom: 0.5rem;">Upload an AI-edited image to analyze</h3>
            <p style="color: #94A3B8; line-height: 1.6;">
                Upload any image that was modified by an AI tool — we'll detect exactly 
                which regions were AI-generated and which parts are from the original photo.
            </p>
            <p style="color: #64748B; font-size: 0.85rem; margin-top: 1rem;">
                Example: You uploaded a selfie to ChatGPT and asked it to change your clothes. 
                This tool will highlight exactly where the clothes were AI-generated.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔬 Forensic Techniques Used</div>',
                   unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔥</div>
                <div class="feature-title">Error Level Analysis</div>
                <div class="feature-desc">Detects inconsistent JPEG compression levels across regions</div>
            </div>
            """, unsafe_allow_html=True)

        with fc2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📡</div>
                <div class="feature-title">Noise Analysis</div>
                <div class="feature-desc">Finds regions where noise patterns differ from the camera sensor</div>
            </div>
            """, unsafe_allow_html=True)

        with fc3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Frequency Analysis</div>
                <div class="feature-desc">Local FFT to detect AI-generated texture fingerprints</div>
            </div>
            """, unsafe_allow_html=True)

        with fc4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔗</div>
                <div class="feature-title">Edge Coherence</div>
                <div class="feature-desc">Detects boundary artifacts between real and AI-edited regions</div>
            </div>
            """, unsafe_allow_html=True)
