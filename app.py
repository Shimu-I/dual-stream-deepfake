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
    page_title="Beyond Binary Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ───────────────────────────────────────────────
# CSS  — refined forensic dark theme
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #080b12;
    --surface:  #0e1320;
    --border:   #1c2236;
    --accent:   #4f8eff;
    --fake:     #ff4466;
    --real:     #00e676;
    --muted:    #3a4060;
    --text:     #cdd5f0;
    --textdim:  #5a6280;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── HEADER BAND ── */
.hdr {
    background: linear-gradient(135deg, #0a0f1e 0%, #0e1528 60%, #111a35 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.4rem 2rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.hdr::before {
    content: '';
    position: absolute;
    top: -40px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(79,142,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hdr-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(79,142,255,0.1);
    border: 1px solid rgba(79,142,255,0.25);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hdr-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.3rem, 2.5vw, 1.9rem);
    font-weight: 700;
    color: #ffffff;
    line-height: 1.35;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hdr-title .hl { color: var(--accent); }
.hdr-sub {
    font-size: 0.82rem;
    color: var(--textdim);
    font-weight: 300;
    letter-spacing: 0.3px;
}
.hdr-sub a { color: var(--accent); text-decoration: none; opacity: 0.8; }
.hdr-sub a:hover { opacity: 1; }
.hdr-pill {
    display: inline-block;
    background: rgba(0,230,118,0.07);
    border: 1px solid rgba(0,230,118,0.18);
    color: var(--real);
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    padding: 2px 8px;
    letter-spacing: 1.5px;
    margin-left: 8px;
    vertical-align: middle;
    text-transform: uppercase;
}

/* ── SECTION LABELS ── */
.slabel {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
}
.slabel::before {
    content: '';
    display: inline-block;
    width: 3px; height: 12px;
    background: var(--accent);
    border-radius: 2px;
}

/* ── VERDICT CARD ── */
.vcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.8rem;
    height: 100%;
}
.vcard-fake { border-left: 3px solid var(--fake); }
.vcard-real { border-left: 3px solid var(--real); }

.vcard-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 3px;
    color: var(--textdim);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.vlabel-fake {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--fake);
    letter-spacing: 8px;
    line-height: 1;
    margin-bottom: 1.4rem;
}
.vlabel-real {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--real);
    letter-spacing: 8px;
    line-height: 1;
    margin-bottom: 1.4rem;
}

.metric-row { margin-bottom: 1.2rem; }
.metric-key {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2.5px;
    color: var(--textdim);
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #e8eeff;
    font-weight: 700;
    line-height: 1;
}
.bar-track {
    background: var(--border);
    border-radius: 3px;
    height: 4px;
    margin-top: 6px;
    overflow: hidden;
}
.bar-fake { background: var(--fake); height: 4px; border-radius: 3px; transition: width 0.8s ease; }
.bar-real { background: var(--real); height: 4px; border-radius: 3px; transition: width 0.8s ease; }

.vfootnote {
    margin-top: 1.4rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    line-height: 1.8;
}

/* ── INFO BOX ── */
.ibox {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: var(--textdim);
    line-height: 1.6;
}

/* ── CAPTION ── */
.cap {
    font-size: 0.7rem;
    color: var(--textdim);
    text-align: center;
    margin-top: 0.6rem;
    line-height: 1.6;
    font-style: italic;
}

/* ── DIVIDER ── */
.divrow {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8rem 0;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid var(--border);
    margin-top: 2.5rem;
    letter-spacing: 1px;
}
.footer a { color: var(--accent); text-decoration: none; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
header {visibility: hidden;}

/* tighten streamlit default padding */
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
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
# SALIENCY  (integrated gradients — more visible than vanilla gradients)
# ───────────────────────────────────────────────
def compute_saliency(model, rgb_in, dct_in, steps=20):
    """
    Integrated gradients from black baseline → input.
    Produces far more visible heatmaps than single-step gradients.
    """
    try:
        baseline   = np.zeros_like(rgb_in)
        alphas     = np.linspace(0, 1, steps)
        grad_acc   = np.zeros_like(rgb_in)

        dct_t = tf.constant(np.expand_dims(dct_in, 0).astype(np.float32))

        for alpha in alphas:
            interp = baseline + alpha * (rgb_in - baseline)
            rgb_t  = tf.Variable(np.expand_dims(interp, 0).astype(np.float32))
            with tf.GradientTape() as tape:
                tape.watch(rgb_t)
                preds = model([rgb_t, dct_t], training=False)
                score = preds[:, 0]
            grads     = tape.gradient(score, rgb_t).numpy()[0]
            grad_acc += grads

        # Integrated gradients attribution
        ig         = (rgb_in - baseline) * grad_acc / steps
        heatmap    = np.sum(np.abs(ig), axis=-1)
        max_val    = float(np.max(heatmap))
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap
    except Exception:
        return None

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=140, facecolor="#080b12")
    buf.seek(0)
    return buf


# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.markdown(
    '<div class="hdr">'
    '  <div class="hdr-badge">🔬 &nbsp; Forensic AI Tool</div>'
    '  <div class="hdr-title">'
    '    Beyond Binary Detection:<br>'
    '    <span class="hl">Explainable AI (XAI)</span> Deepfake Localization<br>'
    '    via Frequency-Aware Segmentation'
    '  </div>'
    '  <div class="hdr-sub" style="margin-top:0.7rem;">'
    '    Dual-stream spatial + frequency domain analysis'
    '    <span class="hdr-pill">CIFAKE · EfficientNetB0</span>'
    '    &nbsp;·&nbsp;'
    '    <a href="https://github.com/Shimu-I/dual-stream-deepfake" target="_blank">'
    '    github.com/Shimu-I/dual-stream-deepfake</a>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True
)


# ───────────────────────────────────────────────
# UPLOAD
# ───────────────────────────────────────────────
st.markdown('<div class="slabel">Step 01 — Upload Image</div>',
            unsafe_allow_html=True)

upload = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if upload is None:
    st.markdown(
        '<div class="ibox">'
        '📂 &nbsp; Upload a JPG or PNG image above to begin forensic analysis. '
        'The dual-stream model will analyse both spatial pixel features and '
        'DCT frequency artifacts simultaneously.'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()


# ───────────────────────────────────────────────
# PREVIEW + RUN BUTTON
# ───────────────────────────────────────────────
prev_col, btn_col = st.columns([2, 1], gap="large")

with prev_col:
    st.markdown('<div class="slabel">Preview</div>', unsafe_allow_html=True)
    preview = Image.open(upload)
    st.image(preview, width=320)

with btn_col:
    st.markdown('<div class="slabel">Step 02 — Run Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="ibox" style="margin-bottom:1.2rem;">'
        'Image loaded successfully.<br><br>'
        'The model will run both streams in parallel and return a '
        'classification, confidence score, saliency heatmap, and '
        'DCT frequency map.'
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
# ROW 1 — Original Image | Verdict
# ───────────────────────────────────────────────
st.markdown('<hr class="divrow">', unsafe_allow_html=True)
col_img, col_verdict = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<div class="slabel">Original Image</div>', unsafe_allow_html=True)
    st.image(pil_img, width=min(pil_img.width, 420))

with col_verdict:
    st.markdown('<div class="slabel">Forensic Verdict</div>', unsafe_allow_html=True)

    icon     = "🚨" if is_fake else "✅"
    lbl_cls  = "vlabel-fake" if is_fake else "vlabel-real"
    card_cls = "vcard-fake"  if is_fake else "vcard-real"
    bar_cls  = "bar-fake"    if is_fake else "bar-real"

    st.markdown(
        f'<div class="vcard {card_cls}">'
        f'  <div class="vcard-tag">Classification Result</div>'
        f'  <div class="{lbl_cls}">{icon}&nbsp; {label}</div>'
        f'  <div class="metric-row">'
        f'    <div class="metric-key">Confidence</div>'
        f'    <div class="metric-val">{bar_pct}%</div>'
        f'    <div class="bar-track">'
        f'      <div class="{bar_cls}" style="width:{bar_pct}%"></div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="metric-row">'
        f'    <div class="metric-key">Raw Model Score</div>'
        f'    <div class="metric-val">{score:.4f}</div>'
        f'  </div>'
        f'  <div class="vfootnote">'
        f'    Score → 0.0 = FAKE &nbsp;|&nbsp; Score → 1.0 = REAL<br>'
        f'    Threshold: 0.5 &nbsp;|&nbsp; FAKE_CLASS_IDX = {FAKE_CLASS_IDX}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ───────────────────────────────────────────────
# ROW 2 — Saliency Heatmap | DCT Spectrum
# ───────────────────────────────────────────────
st.markdown('<hr class="divrow">', unsafe_allow_html=True)
col_heat, col_dct = st.columns([1, 1], gap="large")

with col_heat:
    st.markdown('<div class="slabel">Spatial Attention — Integrated Gradients</div>',
                unsafe_allow_html=True)

    if heatmap is not None:
        h, w       = img_display.shape[:2]
        hm_r       = cv2.resize(heatmap, (w, h))
        hm_s       = cv2.GaussianBlur(hm_r, (9, 9), 0)
        cmap       = plt.get_cmap("RdYlBu_r")
        colored    = (cmap(hm_s)[:, :, :3] * 255).astype(np.uint8)
        overlay    = cv2.addWeighted(img_display.astype(np.uint8), 0.55,
                                     colored, 0.45, 0)

        fig = plt.figure(figsize=(9, 4), facecolor="#080b12")
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.12)

        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(overlay)
        ax0.set_title("Activation Overlay", color="#6070a0",
                      fontsize=8.5, pad=8, fontfamily="monospace")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[1])
        im  = ax1.imshow(hm_s, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax1.set_title("Gradient Attribution Map", color="#6070a0",
                      fontsize=8.5, pad=8, fontfamily="monospace")
        ax1.axis("off")
        cbar = fig.colorbar(im, ax=ax1, fraction=0.045, pad=0.03)
        cbar.ax.tick_params(colors="#3a4060", labelsize=7)
        cbar.outline.set_edgecolor("#1c2236")

        st.image(fig_to_buf(fig), use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<div class="cap">'
            '🔴 Red / Yellow = high activation (regions driving the prediction) &nbsp;·&nbsp;'
            '🔵 Blue = low activation (neutral areas)'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="ibox">⚠️ Saliency map could not be computed for this model configuration.</div>',
            unsafe_allow_html=True
        )

with col_dct:
    st.markdown('<div class="slabel">Frequency Domain — DCT Spectrum</div>',
                unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4.5, 4.5), facecolor="#080b12")
    im2 = ax2.imshow(dct_in[:, :, 0], cmap="inferno", vmin=0, vmax=1)
    ax2.set_title("DCT Coefficient Map", color="#6070a0",
                  fontsize=8.5, pad=8, fontfamily="monospace")
    ax2.axis("off")
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cbar2.ax.tick_params(colors="#3a4060", labelsize=7)
    cbar2.outline.set_edgecolor("#1c2236")
    fig2.tight_layout(pad=1.2)
    st.image(fig_to_buf(fig2), use_container_width=True)
    plt.close(fig2)
    st.markdown(
        '<div class="cap">'
        'Periodic checkerboard patterns concentrated in the top-left quadrant '
        'indicate AI upsampling artifacts — invisible to the human eye but '
        'detectable in frequency space.'
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
    '<a href="https://github.com/Shimu-I/dual-stream-deepfake">'
    'github.com/Shimu-I/dual-stream-deepfake</a>'
    '</div>',
    unsafe_allow_html=True
)