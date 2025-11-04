# app.py
# Streamlit demo for LDCT: Pipeline 1 (direct classify) and Pipeline 2 (denoise -> classify)
# Works with models defined in models.py: UNetDenoiser, ResNetClassifier
# Expects weights in saved_models/{denoiser.pt, classifier.pt}

from pathlib import Path
from io import BytesIO
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# -----------------------
# Config & paths
# -----------------------
ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "saved_models"
DENOISER_WEIGHTS = SAVE / "denoiser.pt"
CLASSIFIER_WEIGHTS = SAVE / "classifier.pt"

st.set_page_config(page_title="LDCT Demo", layout="wide")
st.title("LDCT Stroke Classification Demo")
st.caption("Pipeline 1: Direct classify • Pipeline 2: Denoise → Classify (UNet → ResNet)")

# -----------------------
# Import models from your repo
# -----------------------
try:
    from models import UNetDenoiser, ResNetClassifier
except Exception as e:
    st.error(
        "Could not import models from models.py. "
        "Make sure it defines `UNetDenoiser` and `ResNetClassifier`.\n\n"
        f"Import error: {e}"
    )
    st.stop()

# -----------------------
# Model loaders (cached)
# -----------------------
@st.cache_resource
def load_denoiser():
    if not DENOISER_WEIGHTS.exists():
        return None, "Missing denoiser weights at saved_models/denoiser.pt"
    m = UNetDenoiser()  # your UNet is 1-channel in/out by default
    state = torch.load(str(DENOISER_WEIGHTS), map_location="cpu")
    m.load_state_dict(state, strict=False)
    m.eval()
    return m, None

@st.cache_resource
@st.cache_resource
def load_classifier():
    from pathlib import Path
    dummy_path = str(Path("DO_NOT_USE_BACKBONE_WEIGHTS"))  # non-None
    m = ResNetClassifier(in_channels=1, weights_path=dummy_path)
    state = torch.load("saved_models/classifier.pt", map_location="cpu")
    m.load_state_dict(state, strict=False)
    m.eval()
    return m, None

denoiser, denoiser_err = load_denoiser()
classifier, classifier_err = load_classifier()

# -----------------------
# Helpers
# -----------------------
def to_numpy_gray(file_bytes_or_pil) -> np.ndarray:
    """
    Accepts: .npz (with 'arr_0' or first key), PNG/JPG, or PIL.Image.
    Returns: float32 HxW in [0,1].
    """
    if isinstance(file_bytes_or_pil, (bytes, bytearray)):
        # Try NPZ first
        try:
            with np.load(BytesIO(file_bytes_or_pil)) as npz:
                key = "arr_0" if "arr_0" in npz.files else npz.files[0]
                arr = npz[key].astype("float32")
        except Exception:
            pil = Image.open(BytesIO(file_bytes_or_pil)).convert("L")
            arr = np.array(pil, dtype="float32")
    elif isinstance(file_bytes_or_pil, Image.Image):
        arr = np.array(file_bytes_or_pil.convert("L"), dtype="float32")
    else:
        raise ValueError("Unsupported input type for to_numpy_gray")

    # Normalize to [0,1]
    vmax = float(arr.max())
    vmin = float(arr.min())
    if vmax > 1.0:
        if vmax <= 255.0:
            arr = arr / 255.0
        else:
            rng = max(1e-6, vmax - vmin)
            arr = (arr - vmin) / rng
    arr = np.clip(arr, 0.0, 1.0)
    return arr

def to_classifier_tensor(gray_hw: np.ndarray) -> torch.Tensor:
    """
    HxW [0,1] -> 1x1xHxW float tensor for your 1-channel ResNetClassifier.
    No ImageNet normalization needed (your model is 1-ch).
    """
    x = torch.from_numpy(gray_hw).unsqueeze(0).unsqueeze(0).float().clamp(0, 1)
    return x

def psnr_ssim_simple(ref: np.ndarray, test: np.ndarray):
    """
    Compute PSNR/SSIM for two numpy images in [0,1], HxW.
    Minimal SSIM (not windowed); for exact metrics, use skimage if installed.
    """
    ref = ref.astype("float32")
    test = test.astype("float32")
    # PSNR
    mse = float(np.mean((ref - test) ** 2))
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
    # SSIM (global)
    mu_x, mu_y = ref.mean(), test.mean()
    sigma_x, sigma_y = ref.std(), test.std()
    sigma_xy = float(((ref - mu_x) * (test - mu_y)).mean())
    C1, C2 = (0.01**2), (0.03**2)
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2) + 1e-12
    )
    return float(psnr), float(ssim)

def find_sample(path_candidates):
    """
    Try to auto-discover a sample image from your repo structure.
    Returns bytes or None.
    """
    # Prefer denoised dataset samples
    for root in path_candidates:
        p = Path(root)
        if p.exists():
            # look for denoised npz
            for npz in p.rglob("image.npz"):
                try:
                    return npz.read_bytes()
                except Exception:
                    pass
            # fallback: any .png/.jpg
            for img in list(p.rglob("*.png")) + list(p.rglob("*.jpg")) + list(p.rglob("*.jpeg")):
                try:
                    return img.read_bytes()
                except Exception:
                    pass
    return None

# -----------------------
# UI controls
# -----------------------
colA, colB = st.columns([1, 1])
with colA:
    uploaded = st.file_uploader(
        "Upload a CT slice (.npz with 'arr_0', or .png/.jpg)", type=["npz", "png", "jpg", "jpeg"]
    )
with colB:
    ref_uploaded = st.file_uploader(
        "Optional: Upload reference (clean/high-dose) slice for PSNR/SSIM", type=["npz", "png", "jpg", "jpeg"]
    )

pipeline = st.radio(
    "Choose pipeline",
    ["Pipeline 1 — Direct", "Pipeline 2 — Denoise → Classify"],
    horizontal=True,
)
threshold = st.slider("Decision threshold (hemorrhage=1)", 0.05, 0.95, 0.50, 0.01)

with st.sidebar:
    st.subheader("Options")
    auto_load = st.checkbox("Load a sample slice from repository (if no upload)")
    show_debug = st.checkbox("Debug info")

# -----------------------
# Model availability warnings
# -----------------------
if classifier_err:
    st.warning(classifier_err)
if pipeline.endswith("Denoise → Classify") and denoiser_err:
    st.warning(denoiser_err)

# -----------------------
# Inference button
# -----------------------
run = st.button("Run Inference")

if run:
    # Acquire input bytes
    file_bytes = uploaded.getvalue() if uploaded else None
    if file_bytes is None and auto_load:
        file_bytes = find_sample(
            [
                ROOT / "denoised_dataset" / "test",
                ROOT / "dataset" / "test",
                ROOT / "denoised_dataset",
                ROOT / "dataset",
            ]
        )
        if file_bytes is not None:
            st.info("Loaded a sample from repository (no file uploaded).")
    if file_bytes is None:
        st.error("Please upload an input file or enable 'Load a sample slice'.")
        st.stop()

    # Convert to HxW float [0,1]
    try:
        img = to_numpy_gray(file_bytes)
    except Exception as e:
        st.error(f"Failed to decode input: {e}")
        st.stop()

    H, W = img.shape
    if show_debug:
        st.write({"input_shape": (H, W), "min": float(img.min()), "max": float(img.max()), "mean": float(img.mean())})

    # Default: Pipeline 1 (direct)
    x = to_classifier_tensor(img)
    display_denoised = None

    # Pipeline 2: Denoise first
    if pipeline.endswith("Denoise → Classify"):
        if denoiser is None:
            st.error("Denoiser weights not found. Place saved_models/denoiser.pt or select Pipeline 1.")
            st.stop()
        with torch.no_grad():
            d_in = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().clamp(0, 1)  # 1x1xHxW
            den = denoiser(d_in)
            # If your UNet uses tanh, uncomment the next line:
            # den = (den + 1) * 0.5
            den = den.clamp(0, 1)
            display_denoised = den.squeeze(0).squeeze(0).cpu().numpy()
            x = to_classifier_tensor(display_denoised)

    # Classify
    if classifier is None:
        st.error("Classifier weights not found. Place saved_models/classifier.pt and try again.")
        st.stop()

    with torch.no_grad():
        logits = classifier(x)
        prob = torch.sigmoid(logits).flatten().item()
        pred = int(prob >= threshold)
        # --- Result card ---
        st.markdown("---")
        st.subheader("Prediction")

        label_name = "Hemorrhage" if pred == 1 else "No hemorrhage"
        confidence = prob if pred == 1 else (1 - prob)  # confidence aligned to predicted class
        confidence_pct = f"{100.0 * confidence:.1f}%"

        # Color the badge by class
        badge_color = "#d9534f" if pred == 1 else "#5cb85c"  # red for Hemorrhage, green for No hemorrhage

        st.markdown(
        f"""
         <div style="padding:16px;border:1px solid #eee;border-radius:12px;">
         <div style="font-size:20px;margin-bottom:8px;">
        <strong>Predicted class:</strong>
        <span style="background:{badge_color};color:white;padding:4px 10px;border-radius:999px;margin-left:8px;">
          {label_name}
        </span>
         </div>
        <div style="font-size:16px;margin:6px 0;">
        <strong>Hemorrhage probability:</strong> {prob:.3f}
        &nbsp;&nbsp;|&nbsp;&nbsp;<strong>Decision threshold:</strong> {threshold:.2f}
         </div>
         <div style="font-size:16px;margin:6px 0;">
          <strong>Confidence in predicted class:</strong> {confidence_pct}
         </div>
         </div>
    """,
        unsafe_allow_html=True,
)

# Simple confidence bar (0–1) based on hemorrhage probability
        st.caption("Confidence bar (probability of hemorrhage)")
        st.progress(min(max(prob, 0.0), 1.0))
        # --- Optional interpretation message ---
        msg = (
         " Model flags **hemorrhage** risk above threshold; consider clinical follow-up."
        if pred == 1 else
         " Model estimates **no hemorrhage** at this threshold; treat as a screening signal only."
)  
        st.info(msg)


# Optional debug info toggle already exists; add a touch more if enabled
    if show_debug:
       st.write({
         "prob_hemorrhage": float(prob),
         "pred_label": label_name,
         "confidence_for_pred": float(confidence),
     })


    # Layout: images
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Input")
        st.image(img, use_container_width=True)

    if display_denoised is not None:
        with c2:
            st.subheader("Denoised")
            st.image(display_denoised, clamp=True, use_column_width=True)
    if ref_uploaded is not None:
        try:
            ref = to_numpy_gray(ref_uploaded.getvalue())
            if ref.shape == img.shape:
                PSNR, SSIM = psnr_ssim_simple(ref, display_denoised if display_denoised is not None else img)
                with c3:
                    st.subheader("Quality vs. reference")
                    st.metric("PSNR (dB)", f"{PSNR:.3f}")
                    st.metric("SSIM", f"{SSIM:.3f}")
            else:
                st.info("Reference shape does not match input; skipping PSNR/SSIM.")
        except Exception as e:
            st.info(f"Could not compute PSNR/SSIM: {e}")

    # Prediction block
    st.markdown("---")
    st.subheader("Prediction")
    st.metric("Hemorrhage probability", f"{prob:.3f}")
    st.write(f"Decision threshold = {threshold:.2f} → **Predicted class: {pred}**")

    # Debug info
    if show_debug:
        st.write("Weights present:", {
            "denoiser.pt": DENOISER_WEIGHTS.exists(),
            "classifier.pt": CLASSIFIER_WEIGHTS.exists(),
        })
        st.write("Pipeline:", pipeline)
        if display_denoised is not None:
            st.write({
                "denoised_min": float(display_denoised.min()),
                "denoised_max": float(display_denoised.max()),
                "denoised_mean": float(display_denoised.mean()),
            })

