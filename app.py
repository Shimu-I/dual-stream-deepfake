
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import matplotlib.pyplot as plt
import io


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)


# ===== MODEL LOADING =====
@st.cache_resource
def load_detector():
    return load_model("dual_stream_final.keras")


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
    img_rgb  = np.array(pil_image.convert("RGB"))
    img_128  = cv2.resize(img_rgb, (128, 128))
    rgb_in   = img_128.astype(np.float32) / 255.0
    dct_in   = dct_preprocess(img_rgb)
    return img_128, rgb_in, dct_in


# ===== GRAD-CAM =====
def compute_gradcam(model, rgb_in, dct_in, last_conv_name="top_conv"):
    try:
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(
                [np.expand_dims(rgb_in, 0), np.expand_dims(dct_in, 0)]
            )
            score = preds[:, 0]

        grads        = tape.gradient(score, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out     = conv_out[0]
        heatmap      = (conv_out @ pooled_grads[..., tf.newaxis]).numpy().squeeze()
        heatmap      = np.maximum(heatmap, 0)
        if heatmap.max() != 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception:
        return None


def overlay_heatmap(original, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    colored         = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    colored         = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended         = cv2.addWeighted(original.astype(np.uint8), 1 - alpha, colored, alpha, 0)
    return blended


# ===== UI LAYOUT =====
st.title("🔍 Deepfake Image Detector")
st.markdown(
    "Upload an image to detect whether it is **REAL** or **AI-generated (FAKE)**.  "
    "The model uses a dual-stream architecture combining spatial and frequency-domain analysis."
)

st.divider()

upload = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if upload is not None:
    pil_img = Image.open(upload)

    st.subheader("Uploaded Image")
    st.image(pil_img, width=300)

    with st.spinner("Analysing image..."):
        img_display, rgb_in, dct_in = prepare_inputs(pil_img)

        score = model.predict(
            [np.expand_dims(rgb_in, 0), np.expand_dims(dct_in, 0)], verbose=0
        )[0][0]

        label      = "FAKE" if score >= 0.5 else "REAL"
        confidence = score if score >= 0.5 else 1 - score

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if label == "FAKE":
            st.error(f"🚨 Prediction : **{label}**")
        else:
            st.success(f"✅ Prediction : **{label}**")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.metric("Raw Score (FAKE probability)", f"{score:.4f}")

    with col2:
        heatmap = compute_gradcam(model, rgb_in, dct_in)
        if heatmap is not None:
            overlay = overlay_heatmap(img_display, heatmap)
            st.image(overlay, caption="Grad-CAM — Spatial Attention Map", width=300)
        else:
            st.info("Grad-CAM visualization unavailable for this model configuration.")

    st.divider()
    st.subheader("Frequency Domain View")
    dct_vis = dct_in[:, :, 0]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(dct_vis, cmap="inferno")
    ax.set_title("DCT Frequency Spectrum")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    st.image(buf, width=300)
    plt.close()

else:
    st.info("Upload a JPG or PNG image to begin.")
