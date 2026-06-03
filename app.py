import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Beyond Binary — Deepfake Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ===== CUSTOM CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── Header ── */
.bb-header {
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}

.bb-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #ffffff;
    margin: 0;
}

.bb-title span {
    color: #7c6af7;
}

.bb-subtitle {
    font-size: 0.95rem;
    color: #6b6b80;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7c6af7;
    margin-bottom: 0.5rem;
}

/* ── Result cards ── */
.result-card {
    background: #11111a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
}

.result-card-fake {
    border-color: #ff4757;
    background: linear-gradient(135deg, #1a0a0a, #11111a);
}

.result-card-real {
    border-color: #2ed573;
    background: linear-gradient(135deg, #0a1a0f, #11111a);
}

.verdict-fake {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #ff4757;
    letter-spacing: 4px;
}

.verdict-real {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #2ed573;
    letter-spacing: 4px;
}

.score-label {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b6b80;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
}

.score-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
}

.confidence-bar-bg {
    background: #1e1e2e;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.5rem;
    overflow: hidden;
}

.confidence-bar-fill-fake {
    background: linear-gradient(90deg, #ff4757, #ff6b81);
    height: 6px;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.confidence-bar-fill-real {
    background: linear-gradient(90deg, #2ed573, #7bed9f);
    height: 6px;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7c6af7;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e1e2e;
}

.caption-text {
    font-size: 0.75rem;
    color: #6b6b80;
    text-align: center;
    margin-top: 0.5rem;
}

.info-box {
    background: #11111a;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #7c6af7;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #9090a8;
    margin-top: 1rem;
}

/* ── Hide default streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Divider ── */
hr {
    border-color: #1e1e2e !important;
    margin: 2rem 0 !important;
}

/* ── Image display ── */
.stImage > img {
    border-radius: 10px;
    border: 1px solid #1e1e2e;
}
</style>
""", unsafe_allow_html=True)


# ===== CLASS INDEX =====
# 0 = FAKE is class 0 (Kaggle alphabetical default: FAKE < REAL)
FAKE_CLASS_IDX = 0


# ===== MODEL LOADING =====
@st.cache_resource
def load_detector():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dual_stream_final.keras"
    )
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.stop()
    return load_model(model_path)


model = load_detector()


# ===== PREPROCESSING =====
def dct_preprocess(image):
    image = image.astype(np.uint8)
    gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray  = cv2.resize(gray, (128, 128))
    gray  = np.float32(gray)
    dct   = cv2.dct(gray)
    dct   = np.log(np.abs(dct) + 1)
    dct   = cv2.normalize(dct, None, 0, 1, cv2.NORM_MINMAX)
    dct   = np.stack([dct] * 3, axis=-1)
    return dct


def prepare_inputs(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_128 = cv2.resize(img_rgb, (128, 128))
    rgb_in  = img_128.astype(np.float32) / 255.0
    dct_in  = dct_preprocess(img_rgb)
    return img_128, rgb_in, dct_in


# ===== GRADIENT SALIENCY MAP =====
def compute_saliency(model, rgb_in, dct_in):
    """Gradient-based saliency — works with all model architectures."""
    try:
        rgb_t = tf.Variable(
            np.expand_dims(rgb_in, 0).astype(np.float32)
        )
        dct_t = tf.constant(
            np.expand_dims(dct_in, 0).astype(np.float32)
        )
        with tf.GradientTape() as tape:
            tape.watch(rgb_t)
            preds = model([rgb_t, dct_t], training=False)
            score = preds[:, 0]
        grads   = tape.gradient(score, rgb_t)
        heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
        max_val = float(np.max(heatmap))
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap
    except Exception as e:
        return None


def render_heatmap_overlay(original_img, heatmap, alpha=0.55):
    """
    Renders a high-quality heatmap overlay using a forensic-style colormap.
    Cold (blue) = low activation. Hot (red/yellow) = high activation = suspicious.
    """
    h, w = original_img.shape[:2]
    resized = cv2.resize(heatmap, (w, h))

    # Smooth the heatmap slightly for better visuals
    resized = cv2.GaussianBlur(resized, (11, 11), 0)

    # Use a forensic-style colormap: dark blue → cyan → yellow → red
    colormap = plt.get_cmap('RdYlBu_r')
    colored  = (colormap(resized)[:, :, :3] * 255).astype(np.uint8)

    # Blend with original
    overlay = cv2.addWeighted(
        original_img.astype(np.uint8), 1 - alpha,
        colored, alpha, 0
    )
    return overlay, resized


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=130, facecolor="#0a0a0f")
    buf.seek(0)
    return buf


# ===== HEADER =====
st.markdown("""
<div class="bb-header">
    <p class="bb-title">🔬 Beyond <span>Binary</span></p>
    <p class="bb-subtitle">
        Deepfake forensics via dual-stream spatial + frequency analysis &nbsp;·&nbsp;
        Upload an image to inspect its mathematical fingerprint
    </p>
</div>
""", unsafe_allow_html=True)


# ===== UPLOAD =====
st.markdown('<p class="upload-label">▸ Input Image</p>', unsafe_allow_html=True)
upload = st.file_uploader("", type=["jpg", "jpeg", "png"],
                           label_visibility="collapsed")

if upload is None:
    st.markdown("""
    <div class="info-box">
        📂 &nbsp; Drop a JPG or PNG image above to begin forensic analysis.
        The system will analyse both the <strong>spatial (RGB)</strong> and
        <strong>frequency (DCT)</strong> domains simultaneously.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ===== PROCESS =====
pil_img = Image.open(upload)

with st.spinner("Running dual-stream analysis…"):
    img_display, rgb_in, dct_in = prepare_inputs(pil_img)

    score = float(model.predict(
        [np.expand_dims(rgb_in, 0), np.expand_dims(dct_in, 0)],
        verbose=0
    )[0][0])

    is_fake    = (score < 0.5) if FAKE_CLASS_IDX == 0 else (score >= 0.5)
    label      = "FAKE" if is_fake else "REAL"
    confidence = (1.0 - score) if (FAKE_CLASS_IDX == 0 and is_fake) else score
    confidence = max(confidence, 1.0 - confidence)

    heatmap = compute_saliency(model, rgb_in, dct_in)


# ===== LAYOUT: Row 1 — Image + Verdict =====
st.markdown("---")
col_img, col_verdict = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<p class="section-label">Original Image</p>',
                unsafe_allow_html=True)
    display_w = min(pil_img.width, 500)
    st.image(pil_img, width=display_w)

with col_verdict:
    st.markdown('<p class="section-label">Forensic Verdict</p>',
                unsafe_allow_html=True)

    card_class = "result-card-fake" if is_fake else "result-card-real"
    verdict_class = "verdict-fake" if is_fake else "verdict-real"
    icon = "🚨" if is_fake else "✅"
    bar_class = "confidence-bar-fill-fake" if is_fake else "confidence-bar-fill-real"
    bar_pct = int(confidence * 100)

    st.markdown(f"""
    <div class="result-card {card_class}">
        <div style="font-size:0.75rem; letter-spacing:3px; text-transform:uppercase;
                    color:#6b6b80; margin-bottom:0.5rem;">Classification</div>
        <div class="{verdict_class}">{icon} &nbsp; {label}</div>

        <div class="score-label">Confidence</div>
        <div class="score-value">{bar_pct}%</div>
        <div class="confidence-bar-bg">
            <div class="{bar_class}" style="width:{bar_pct}%"></div>
        </div>

        <div class="score-label">Raw Model Score</div>
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem;
                    color:#9090a8;">{score:.4f}</div>

        <div style="margin-top:1.2rem; font-size:0.75rem; color:#6b6b80;
                    border-top:1px solid #1e1e2e; padding-top:0.8rem;">
            Score → 0 = FAKE &nbsp;|&nbsp; Score → 1 = REAL &nbsp;(FAKE_CLASS_IDX={FAKE_CLASS_IDX})
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===== LAYOUT: Row 2 — Heatmap + DCT =====
st.markdown("---")
col_heat, col_dct = st.columns([1, 1], gap="large")

with col_heat:
    st.markdown('<p class="section-label">Spatial Attention Heatmap</p>',
                unsafe_allow_html=True)

    if heatmap is not None:
        overlay, hm_smooth = render_heatmap_overlay(img_display, heatmap)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4),
                                  facecolor="#0a0a0f")

        # Left: overlay
        axes[0].imshow(overlay)
        axes[0].set_title("Activation Overlay", color="#9090a8",
                           fontsize=9, pad=8)
        axes[0].axis("off")

        # Right: pure heatmap with colorbar
        im = axes[1].imshow(hm_smooth, cmap="RdYlBu_r", vmin=0, vmax=1)
        axes[1].set_title("Gradient Magnitude", color="#9090a8",
                           fontsize=9, pad=8)
        axes[1].axis("off")
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#6b6b80")
        cbar.outline.set_edgecolor("#1e1e2e")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#6b6b80", fontsize=7)

        fig.tight_layout(pad=1.5)
        st.image(fig_to_pil(fig), use_container_width=True)
        plt.close(fig)

        st.markdown("""
        <p class="caption-text">
            🔴 Red / Yellow = high activation (suspicious region) &nbsp;·&nbsp;
            🔵 Blue = low activation (natural region)
        </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            ⚠️ Saliency map could not be computed for this image.
        </div>
        """, unsafe_allow_html=True)

with col_dct:
    st.markdown('<p class="section-label">Frequency Domain (DCT Spectrum)</p>',
                unsafe_allow_html=True)

    fig2, ax = plt.subplots(figsize=(4, 4), facecolor="#0a0a0f")
    dct_vis = dct_in[:, :, 0]
    im2 = ax.imshow(dct_vis, cmap="inferno", vmin=0, vmax=1)
    ax.set_title("DCT Coefficient Map", color="#9090a8", fontsize=9, pad=8)
    ax.axis("off")
    cbar2 = fig2.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_tick_params(color="#6b6b80")
    cbar2.outline.set_edgecolor("#1e1e2e")
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#6b6b80", fontsize=7)
    fig2.tight_layout(pad=1.5)
    st.image(fig_to_pil(fig2), use_container_width=True)
    plt.close(fig2)

    st.markdown("""
    <p class="caption-text">
        Periodic grid patterns (checkerboard) in the top-left corner indicate
        AI upsampling artifacts invisible to the human eye.
    </p>
    """, unsafe_allow_html=True)


# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#3a3a50;
            font-family:'Space Mono',monospace; padding:1rem 0;">
    BEYOND BINARY &nbsp;·&nbsp; Dual-Stream Deepfake Forensics &nbsp;·&nbsp;
    Spatial + Frequency Domain Analysis
</div>
""", unsafe_allow_html=True)
