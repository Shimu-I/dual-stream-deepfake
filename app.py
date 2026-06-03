import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io


# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Beyond Binary Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────────────────────────────
# CSS
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0d14;
    color: #c8c8d8;
}
.header-wrap {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1f1f30;
    margin-bottom: 2rem;
}
.header-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    color: #5a5aff;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.4;
    margin: 0;
}
.header-title em {
    color: #5a5aff;
    font-style: normal;
}
.header-sub {
    font-size: 0.85rem;
    color: #55556a;
    margin-top: 0.5rem;
    font-weight: 300;
}
.header-sub a { color: #5a5aff; text-decoration: none; }
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #5a5aff;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1f1f30;
}
.verdict-wrap {
    background: #11111c;
    border: 1px solid #1f1f30;
    border-radius: 10px;
    padding: 1.6rem;
}
.verdict-fake { border-left: 4px solid #ff3c5a !important; }
.verdict-real { border-left: 4px solid #23d160 !important; }
.v-classification {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #55556a;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.v-label-fake {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #ff3c5a;
    letter-spacing: 6px;
}
.v-label-real {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #23d160;
    letter-spacing: 6px;
}
.v-row { margin-top: 1.2rem; }
.v-meta {
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #55556a;
    margin-bottom: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
}
.v-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    color: #e0e0f0;
    font-weight: 600;
}
.bar-bg {
    background: #1f1f30;
    border-radius: 3px;
    height: 5px;
    margin-top: 0.4rem;
    overflow: hidden;
}
.bar-fake { background: #ff3c5a; height: 5px; border-radius: 3px; }
.bar-real { background: #23d160; height: 5px; border-radius: 3px; }
.v-hint {
    margin-top: 1.2rem;
    font-size: 0.7rem;
    color: #35354a;
    border-top: 1px solid #1f1f30;
    padding-top: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
}
.cap {
    font-size: 0.73rem;
    color: #55556a;
    text-align: center;
    margin-top: 0.5rem;
}
.info-box {
    background: #11111c;
    border: 1px solid #1f1f30;
    border-left: 3px solid #5a5aff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #7070a0;
}
.footer {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #2a2a40;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid #1f1f30;
    margin-top: 2rem;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────
# CLASS INDEX
# ───────────────────────────────────────────────
FAKE_CLASS_IDX = 0


# ───────────────────────────────────────────────
# MODEL
# ───────────────────────────────────────────────
@st.cache_resource
def load_detector():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dual_stream_final.keras"
    )
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    return load_model(model_path)

model = load_detector()


# ───────────────────────────────────────────────
# PREPROCESSING
# ───────────────────────────────────────────────
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


# ───────────────────────────────────────────────
# GRADIENT SALIENCY
# ───────────────────────────────────────────────
def compute_saliency(model, rgb_in, dct_in):
    try:
        rgb_t = tf.Variable(np.expand_dims(rgb_in, 0).astype(np.float32))
        dct_t = tf.constant(np.expand_dims(dct_in, 0).astype(np.float32))
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
    except Exception:
        return None

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=130, facecolor="#0d0d14")
    buf.seek(0)
    return buf


# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="header-wrap">'
    '  <div class="header-tag">🔬 &nbsp; Forensic AI Tool</div>'
    '  <div class="header-title">'
    '    Beyond Binary Detection:<br>'
    '    <em>Explainable AI (XAI)</em> Deepfake Localization<br>'
    '    via Frequency-Aware Segmentation'
    '  </div>'
    '  <div class="header-sub">'
    '    Dual-stream spatial + frequency domain analysis &nbsp;·&nbsp;'
    '    <a href="https://github.com/Shimu-I/dual-stream-deepfake" target="_blank">'
    '    github.com/Shimu-I/dual-stream-deepfake</a>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True
)


# ───────────────────────────────────────────────
# UPLOAD
# ───────────────────────────────────────────────
st.markdown('<div class="sec-label">Step 1 — Upload Image</div>',
            unsafe_allow_html=True)

upload = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if upload is None:
    st.markdown(
        '<div class="info-box">'
        '📂 &nbsp; Upload a JPG or PNG image above, then click '
        '<strong>Analyse Image</strong> to run the forensic analysis.'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ───────────────────────────────────────────────
# PREVIEW + CONFIRM BUTTON
# ───────────────────────────────────────────────
prev_col, btn_col = st.columns([2, 1], gap="large")

with prev_col:
    st.markdown('<div class="sec-label">Preview</div>', unsafe_allow_html=True)
    preview = Image.open(upload)
    st.image(preview, width=300)

with btn_col:
    st.markdown('<div class="sec-label">Step 2 — Run Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box" style="margin-bottom:1rem;">'
        'Image loaded successfully.<br><br>'
        'Click <strong>Analyse Image</strong> to run the '
        'dual-stream deepfake detector.'
        '</div>',
        unsafe_allow_html=True
    )
    run = st.button("🔍  Analyse Image", use_container_width=True,
                    type="primary")

if not run:
    st.stop()


# ───────────────────────────────────────────────
# INFERENCE
# ───────────────────────────────────────────────
with st.spinner("Running dual-stream analysis…"):
    pil_img = Image.open(upload)
    img_display, rgb_in, dct_in = prepare_inputs(pil_img)

    score = float(model.predict(
        [np.expand_dims(rgb_in, 0), np.expand_dims(dct_in, 0)],
        verbose=0
    )[0][0])

    is_fake    = (score < 0.5) if FAKE_CLASS_IDX == 0 else (score >= 0.5)
    label      = "FAKE" if is_fake else "REAL"
    confidence = (1.0 - score) if (FAKE_CLASS_IDX == 0 and is_fake) else score
    confidence = max(confidence, 1.0 - confidence)
    bar_pct    = int(confidence * 100)
    heatmap    = compute_saliency(model, rgb_in, dct_in)


# ───────────────────────────────────────────────
# ROW 1 — Image | Verdict
# ───────────────────────────────────────────────
st.divider()
col_img, col_verdict = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<div class="sec-label">Original Image</div>',
                unsafe_allow_html=True)
    st.image(pil_img, width=min(pil_img.width, 420))

with col_verdict:
    st.markdown('<div class="sec-label">Forensic Verdict</div>',
                unsafe_allow_html=True)

    icon        = "🚨" if is_fake else "✅"
    lbl_cls     = "v-label-fake" if is_fake else "v-label-real"
    wrap_cls    = "verdict-fake" if is_fake else "verdict-real"
    bar_cls     = "bar-fake"     if is_fake else "bar-real"
    score_str   = f"{score:.4f}"
    conf_str    = f"{bar_pct}%"

    st.markdown(
        f'<div class="verdict-wrap {wrap_cls}">'
        f'<div class="v-classification">Classification</div>'
        f'<div class="{lbl_cls}">{icon} &nbsp; {label}</div>'
        f'<div class="v-row">'
        f'<div class="v-meta">Confidence</div>'
        f'<div class="v-value">{conf_str}</div>'
        f'<div class="bar-bg"><div class="{bar_cls}" style="width:{bar_pct}%"></div></div>'
        f'</div>'
        f'<div class="v-row">'
        f'<div class="v-meta">Raw Model Score</div>'
        f'<div class="v-value">{score_str}</div>'
        f'</div>'
        f'<div class="v-hint">'
        f'Score 0 = FAKE &nbsp;|&nbsp; Score 1 = REAL'
        f'&nbsp; (FAKE_CLASS_IDX = {FAKE_CLASS_IDX})'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ───────────────────────────────────────────────
# ROW 2 — Heatmap | DCT
# ───────────────────────────────────────────────
st.divider()
col_heat, col_dct = st.columns([1, 1], gap="large")

with col_heat:
    st.markdown('<div class="sec-label">Spatial Attention Heatmap</div>',
                unsafe_allow_html=True)

    if heatmap is not None:
        h, w       = img_display.shape[:2]
        hm_resized = cv2.resize(heatmap, (w, h))
        hm_smooth  = cv2.GaussianBlur(hm_resized, (11, 11), 0)
        cmap       = plt.get_cmap("RdYlBu_r")
        colored    = (cmap(hm_smooth)[:, :, :3] * 255).astype(np.uint8)
        overlay    = cv2.addWeighted(
            img_display.astype(np.uint8), 0.5, colored, 0.5, 0
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="#0d0d14")
        fig.subplots_adjust(wspace=0.15)

        axes[0].imshow(overlay)
        axes[0].set_title("Activation Overlay", color="#7070a0",
                          fontsize=9, pad=8, fontfamily="monospace")
        axes[0].axis("off")

        im = axes[1].imshow(hm_smooth, cmap="RdYlBu_r", vmin=0, vmax=1)
        axes[1].set_title("Gradient Magnitude", color="#7070a0",
                          fontsize=9, pad=8, fontfamily="monospace")
        axes[1].axis("off")
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#55556a", labelsize=7)
        cbar.outline.set_edgecolor("#1f1f30")

        st.image(fig_to_buf(fig), use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<div class="cap">'
            '🔴 Red/Yellow = high activation (suspicious) &nbsp;·&nbsp;'
            '🔵 Blue = low activation (natural)'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-box">⚠️ Saliency map could not be computed.</div>',
            unsafe_allow_html=True
        )

with col_dct:
    st.markdown('<div class="sec-label">Frequency Domain — DCT Spectrum</div>',
                unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4, 4), facecolor="#0d0d14")
    im2 = ax2.imshow(dct_in[:, :, 0], cmap="inferno", vmin=0, vmax=1)
    ax2.set_title("DCT Coefficient Map", color="#7070a0",
                  fontsize=9, pad=8, fontfamily="monospace")
    ax2.axis("off")
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#55556a", labelsize=7)
    cbar2.outline.set_edgecolor("#1f1f30")
    fig2.tight_layout(pad=1.5)
    st.image(fig_to_buf(fig2), use_container_width=True)
    plt.close(fig2)
    st.markdown(
        '<div class="cap">'
        'Periodic checkerboard patterns in the top-left corner '
        'indicate AI upsampling artifacts invisible to the human eye.'
        '</div>',
        unsafe_allow_html=True
    )


# ───────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Beyond Binary Detection &nbsp;·&nbsp; '
    'XAI Deepfake Localization via Frequency-Aware Segmentation &nbsp;·&nbsp; '
    '<a href="https://github.com/Shimu-I/dual-stream-deepfake" '
    'style="color:#5a5aff; text-decoration:none;">'
    'github.com/Shimu-I/dual-stream-deepfake</a>'
    '</div>',
    unsafe_allow_html=True
)