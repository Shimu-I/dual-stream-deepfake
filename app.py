import os
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io


# ───────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────────────────────────────
# CSS  — clean, minimal, readable
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #58a6ff;
    --fake:    #f85149;
    --real:    #3fb950;
    --text:    #e6edf3;
    --dim:     #8b949e;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-size: 16px;
}

/* hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding: 2rem 2rem 2rem 2rem !important; max-width: 1100px; }

/* ── PAGE HEADER ── */
.page-header {
    padding: 2rem 0 2.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.5rem 0;
    line-height: 1.3;
}
.page-sub {
    font-size: 1rem;
    color: var(--dim);
    margin: 0;
    line-height: 1.5;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── INFO BOX ── */
.info-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 1rem;
    color: var(--dim);
    line-height: 1.7;
}

/* ── VERDICT CARD ── */
.verdict-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 2rem;
}
.verdict-card.fake { border-left: 4px solid var(--fake); }
.verdict-card.real { border-left: 4px solid var(--real); }

.verdict-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

.verdict-result-fake {
    font-size: 3rem;
    font-weight: 700;
    color: var(--fake);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 2rem;
    line-height: 1;
}
.verdict-result-real {
    font-size: 3rem;
    font-weight: 700;
    color: var(--real);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 2rem;
    line-height: 1;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--dim);
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 0.5rem;
    line-height: 1;
}
.bar-track {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    margin-bottom: 1.5rem;
    overflow: hidden;
}
.bar-fill-fake { background: var(--fake); height: 6px; border-radius: 4px; }
.bar-fill-real { background: var(--real); height: 6px; border-radius: 4px; }

.verdict-footnote {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--dim);
    line-height: 1.9;
    border-top: 1px solid var(--border);
    padding-top: 1.25rem;
    margin-top: 0.5rem;
}

/* ── CAPTION ── */
.img-caption {
    font-size: 0.85rem;
    color: var(--dim);
    text-align: center;
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* ── DIVIDER ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

/* ── FOOTER ── */
.page-footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--dim);
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
.page-footer a { color: var(--accent); text-decoration: none; }
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
# SALIENCY
# ───────────────────────────────────────────────
def compute_saliency(model, rgb_in, dct_in, steps=5):
    try:
        noise_level = 0.1
        grad_acc    = np.zeros(rgb_in.shape[:2], dtype=np.float32)
        dct_t       = tf.constant(
            np.expand_dims(dct_in, 0).astype(np.float32)
        )
        for _ in range(steps):
            noise  = np.random.normal(0, noise_level, rgb_in.shape).astype(np.float32)
            rgb_t  = tf.Variable(
                np.expand_dims(rgb_in + noise, 0).astype(np.float32)
            )
            with tf.GradientTape() as tape:
                tape.watch(rgb_t)
                preds = model([rgb_t, dct_t], training=False)
                score = preds[:, 0]
            grads     = tape.gradient(score, rgb_t).numpy()[0]
            grad_acc += np.max(np.abs(grads), axis=-1)

        max_val = float(np.max(grad_acc))
        if max_val > 0:
            grad_acc = grad_acc / max_val
        return grad_acc
    except Exception:
        return None

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=140, facecolor="#0d1117")
    buf.seek(0)
    return buf


# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '  <h1 class="page-title">🔬 Deepfake Detector</h1>'
    '  <p class="page-sub">'
    '    Dual-stream spatial + frequency domain analysis &nbsp;·&nbsp; '
    '    EfficientNetB0 + DCT &nbsp;·&nbsp; '
    '    <a href="https://github.com/Shimu-I/dual-stream-deepfake" style="color:#58a6ff;">'
    '    github.com/Shimu-I/dual-stream-deepfake</a>'
    '  </p>'
    '</div>',
    unsafe_allow_html=True
)


# ───────────────────────────────────────────────
# UPLOAD
# ───────────────────────────────────────────────
st.markdown('<div class="sec-label">Step 1 — Upload Image</div>', unsafe_allow_html=True)

upload = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if upload is None:
    st.markdown(
        '<div class="info-box">'
        '📂 &nbsp; Upload a JPG or PNG above to begin analysis. '
        'The model analyses both pixel-level features and DCT frequency artifacts.'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ───────────────────────────────────────────────
# PREVIEW + RUN
# ───────────────────────────────────────────────
prev_col, btn_col = st.columns([2, 1], gap="large")

with prev_col:
    st.markdown('<div class="sec-label">Preview</div>', unsafe_allow_html=True)
    preview = Image.open(upload)
    st.image(preview, width=340)

with btn_col:
    st.markdown('<div class="sec-label">Step 2 — Run Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box" style="margin-bottom:1.25rem;">'
        'Image loaded. Click the button to run both model streams and get a '
        'classification, confidence score, saliency heatmap, and DCT frequency map.'
        '</div>',
        unsafe_allow_html=True
    )
    run = st.button("🔍  Analyse Image", use_container_width=True, type="primary")

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
st.markdown('<hr class="divider">', unsafe_allow_html=True)
col_img, col_verdict = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<div class="sec-label">Original Image</div>', unsafe_allow_html=True)
    st.image(pil_img, width=min(pil_img.width, 420))

with col_verdict:
    st.markdown('<div class="sec-label">Result</div>', unsafe_allow_html=True)

    icon     = "🚨" if is_fake else "✅"
    res_cls  = "verdict-result-fake" if is_fake else "verdict-result-real"
    card_cls = "fake" if is_fake else "real"
    bar_cls  = "bar-fill-fake" if is_fake else "bar-fill-real"

    st.markdown(
        f'<div class="verdict-card {card_cls}">'
        f'  <div class="verdict-label">Classification</div>'
        f'  <div class="{res_cls}">{icon} {label}</div>'
        f'  <div class="metric-label">Confidence</div>'
        f'  <div class="metric-value">{bar_pct}%</div>'
        f'  <div class="bar-track">'
        f'    <div class="{bar_cls}" style="width:{bar_pct}%"></div>'
        f'  </div>'
        f'  <div class="metric-label">Raw Model Score</div>'
        f'  <div class="metric-value">{score:.4f}</div>'
        f'  <div class="verdict-footnote">'
        f'    Score → 0.0 = FAKE &nbsp;|&nbsp; Score → 1.0 = REAL<br>'
        f'    Threshold: 0.5 &nbsp;|&nbsp; FAKE_CLASS_IDX = {FAKE_CLASS_IDX}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ───────────────────────────────────────────────
# ROW 2 — Saliency | DCT
# ───────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
col_heat, col_dct = st.columns([1, 1], gap="large")

with col_heat:
    st.markdown('<div class="sec-label">Spatial Attention — Saliency Map</div>',
                unsafe_allow_html=True)

    if heatmap is not None:
        h, w    = img_display.shape[:2]
        hm_r    = cv2.resize(heatmap, (w, h))
        hm_s    = cv2.GaussianBlur(hm_r, (9, 9), 0)
        cmap    = plt.get_cmap("RdYlBu_r")
        colored = (cmap(hm_s)[:, :, :3] * 255).astype(np.uint8)
        overlay = cv2.addWeighted(img_display.astype(np.uint8), 0.55,
                                  colored, 0.45, 0)

        fig = plt.figure(figsize=(9, 4), facecolor="#0d1117")
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(overlay)
        ax0.set_title("Activation Overlay", color="#8b949e",
                      fontsize=10, pad=8, fontfamily="monospace")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[1])
        im  = ax1.imshow(hm_s, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax1.set_title("Gradient Attribution", color="#8b949e",
                      fontsize=10, pad=8, fontfamily="monospace")
        ax1.axis("off")
        cbar = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(colors="#8b949e", labelsize=8)
        cbar.outline.set_edgecolor("#30363d")

        st.image(fig_to_buf(fig), use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<div class="img-caption">'
            '🔴 Red/Yellow = high activation (drives the prediction) &nbsp;·&nbsp; 🔵 Blue = low activation'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-box">⚠️ Saliency map could not be computed for this model.</div>',
            unsafe_allow_html=True
        )

with col_dct:
    st.markdown('<div class="sec-label">Frequency Domain — DCT Spectrum</div>',
                unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5), facecolor="#0d1117")
    im2 = ax2.imshow(dct_in[:, :, 0], cmap="inferno", vmin=0, vmax=1)
    ax2.set_title("DCT Coefficient Map", color="#8b949e",
                  fontsize=10, pad=8, fontfamily="monospace")
    ax2.axis("off")
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cbar2.ax.tick_params(colors="#8b949e", labelsize=8)
    cbar2.outline.set_edgecolor("#30363d")
    fig2.tight_layout(pad=1.2)
    st.image(fig_to_buf(fig2), use_container_width=True)
    plt.close(fig2)
    st.markdown(
        '<div class="img-caption">'
        'Checkerboard patterns in the top-left quadrant indicate AI upsampling artifacts.'
        '</div>',
        unsafe_allow_html=True
    )


# ───────────────────────────────────────────────
# FOOTER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="page-footer">'
    'Beyond Binary Detection &nbsp;·&nbsp; XAI Deepfake Localization &nbsp;·&nbsp; '
    '<a href="https://github.com/Shimu-I/dual-stream-deepfake">'
    'github.com/Shimu-I/dual-stream-deepfake</a>'
    '</div>',
    unsafe_allow_html=True
)