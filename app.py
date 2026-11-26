import html
import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG = 224
INCONCLUSIVE_LOW = 0.35
INCONCLUSIVE_HIGH = 0.65
HIGH_RISK_THRESHOLD = 0.85

# Exact evaluation metrics from Parkinson.ipynb (60 unseen test drawings)
MODEL_BENCHMARKS = [
    {
        "name": "VGG16 (Active Model)",
        "tag": "Best Overall",
        "test_acc": 81.7,
        "val_acc": 82.8,
        "precision": 78.8,
        "recall": 86.7,
        "f1": 82.5,
        "specificity": 76.7,
        "auc": 0.914,
        "ci": "70.0% - 91.7%",
        "cm": [[23, 7], [4, 26]],
        "highlight": True,
    },
    {
        "name": "VGG16 + ResNet50 Ensemble",
        "tag": "Runner-up",
        "test_acc": 80.0,
        "val_acc": None,
        "precision": None,
        "recall": 80.0,
        "f1": 80.0,
        "specificity": None,
        "auc": 0.903,
        "ci": "70.0% - 90.0%",
        "cm": None,
        "highlight": False,
    },
    {
        "name": "ResNet50 (Fine-tuned)",
        "tag": "Pretrained",
        "test_acc": 75.0,
        "val_acc": 75.9,
        "precision": 75.9,
        "recall": 73.3,
        "f1": 74.6,
        "specificity": 76.7,
        "auc": 0.894,
        "ci": "63.3% - 85.0%",
        "cm": [[23, 7], [8, 22]],
        "highlight": False,
    },
    {
        "name": "Custom CNN (From Scratch)",
        "tag": "Baseline",
        "test_acc": 50.0,
        "val_acc": 48.3,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "specificity": 100.0,
        "auc": 0.524,
        "ci": "36.7% - 63.3%",
        "cm": [[30, 0], [30, 0]],
        "highlight": False,
    },
]

st.set_page_config(
    page_title="Parkinson's Drawing Screening",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    """Inject clean clinical light CSS styling."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.2rem !important;
}

.block-container,
.stMainBlockContainer,
[data-testid="stMainBlockContainer"],
section.main .block-container {
    max-width: 100% !important;
    width: 100% !important;
    padding-top: 0.75rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 2.2rem !important;
}
@media (max-width: 768px) {
    .block-container,
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"],
    section.main .block-container {
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }
}

/* Top Hero */
.hero-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0FDFA 55%, #ECFEFF 100%);
    border: 1px solid #CCFBF1;
    border-radius: 18px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(15, 118, 110, 0.06);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: #CCFBF1;
    color: #115E59;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin: 0 0 0.45rem 0;
}
.hero-sub {
    font-size: 1.02rem;
    color: #475569;
    line-height: 1.55;
    max-width: 100%;
    margin: 0;
}

/* 3-Step Strip */
.step-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-bottom: 1.35rem;
}
@media (max-width: 768px) {
    .step-strip { grid-template-columns: 1fr; }
}
.step-pill {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
.step-num {
    flex-shrink: 0;
    width: 1.85rem;
    height: 1.85rem;
    border-radius: 50%;
    background: #0F766E;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.9rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.step-text strong {
    display: block;
    font-size: 0.93rem;
    color: #0F172A;
    font-weight: 700;
}
.step-text span {
    font-size: 0.83rem;
    color: #64748B;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #F1F5F9;
    padding: 0.35rem;
    border-radius: 12px;
    margin-bottom: 1.1rem;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs button[data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 0.5rem 1.15rem !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    color: #475569 !important;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F766E !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
}

/* Uploader file list cleanliness for multi-file batches */
[data-testid="stFileUploader"] ul {
    max-height: 155px;
    overflow-y: auto;
    padding-top: 0.35rem;
}

/* Upload Zone */
.section-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 0.2rem 0;
}
.section-caption {
    font-size: 0.92rem;
    color: #64748B;
    margin-bottom: 0.75rem;
}

/* Batch Summary */
.batch-bar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0 1.2rem 0;
}
.batch-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #0F172A;
}
.batch-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.mini-pill {
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
}
.mini-pill.warn { background: #FEF3C7; color: #92400E; }
.mini-pill.ok { background: #DCFCE7; color: #166534; }
.mini-pill.unsure { background: #E0E7FF; color: #3730A3; }

/* Result Card Header */
.result-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    padding-bottom: 0.85rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #E2E8F0;
}
.file-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.86rem;
    font-weight: 600;
    color: #334155;
    background: #F1F5F9;
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
}
.status-badge.warn { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
.status-badge.ok { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
.status-badge.unsure { background: #E0E7FF; color: #3730A3; border: 1px solid #C7D2FE; }

/* Verdict & Chance Box */
.chance-card {
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
}
.chance-card.warn { background: #FFFBEB; border: 1px solid #FDE68A; }
.chance-card.ok { background: #F0FDF4; border: 1px solid #BBF7D0; }
.chance-card.unsure { background: #EEF2FF; border: 1px solid #C7D2FE; }

.chance-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}
.chance-label {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
}
.chance-big {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0.2rem 0;
}
.chance-big.warn { color: #B45309; }
.chance-big.ok { color: #15803D; }
.chance-big.unsure { color: #4338CA; }

.risk-tag {
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.25rem 0.65rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.85);
    color: #0F172A;
    border: 1px solid rgba(15, 23, 42, 0.08);
}

/* Probability Bars */
.prob-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    font-weight: 600;
    color: #1E293B;
    margin: 0.7rem 0 0.28rem 0;
}
.prob-track {
    height: 12px;
    background: #E2E8F0;
    border-radius: 999px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
}

/* Explanation & Disclaimer */
.explain-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.95rem 1.05rem;
    margin-top: 0.9rem;
    font-size: 0.93rem;
    line-height: 1.55;
    color: #334155;
}
.explain-box b { color: #0F172A; }

.cam-legend {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-top: 0.45rem;
    font-size: 0.8rem;
    color: #64748B;
}
.legend-bar {
    flex: 1;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(14,165,233,0.18) 0%, #14B8A6 35%, #F59E0B 70%, #EF4444 100%);
}

/* Empty State Tips */
.tips-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1rem;
}
@media (max-width: 768px) {
    .tips-grid { grid-template-columns: 1fr; }
}
.tip-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1rem 1.1rem;
}
.tip-icon { font-size: 1.5rem; margin-bottom: 0.35rem; }
.tip-title { font-weight: 700; font-size: 0.95rem; color: #0F172A; margin-bottom: 0.2rem; }
.tip-desc { font-size: 0.86rem; color: #64748B; line-height: 1.45; }

/* Benchmark Cards & Bars */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.85rem;
    margin: 0.75rem 0 1.35rem 0;
}
@media (max-width: 768px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.1rem 1.15rem;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.85rem;
    font-weight: 800;
    color: #0F766E;
    line-height: 1.1;
}
.kpi-lbl {
    font-size: 0.86rem;
    color: #64748B;
    margin-top: 0.3rem;
    line-height: 1.35;
}

.bench-row {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 0.75rem;
}
.bench-row.active {
    border-color: #5EEAD4;
    background: #F0FDFA;
}
.bench-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.55rem;
}
.bench-name { font-weight: 700; font-size: 0.98rem; color: #0F172A; }
.bench-tag {
    font-size: 0.76rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: #E2E8F0;
    color: #334155;
}
.bench-row.active .bench-tag { background: #0F766E; color: #FFFFFF; }
.bench-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
    font-size: 0.87rem;
    color: #475569;
    margin-top: 0.45rem;
}
.bench-metrics b { color: #0F172A; font-family: 'JetBrains Mono', monospace; }

.cm-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem;
    max-width: 380px;
    margin-top: 0.6rem;
}
.cm-cell {
    border-radius: 12px;
    padding: 0.85rem;
    text-align: center;
    border: 1px solid #E2E8F0;
}
.cm-cell.good { background: #DCFCE7; border-color: #BBF7D0; color: #166534; }
.cm-cell.bad { background: #FEF3C7; border-color: #FDE68A; color: #92400E; }
.cm-num { font-family: 'JetBrains Mono', monospace; font-size: 1.45rem; font-weight: 800; }
.cm-cap { font-size: 0.78rem; font-weight: 600; margin-top: 0.15rem; }

.medical-banner {
    background: #F8FAFC;
    border-left: 4px solid #0F766E;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-top: 1.1rem;
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.5;
}
.app-footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.83rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E2E8F0;
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading Parkinson's VGG16 screening model...")
def load_model():
    """Load VGG16 weights from Hugging Face and prepare Grad-CAM extractor."""
    path = hf_hub_download("saiaditya05/cheyali-vgg16", "vgg16.weights.h5")
    base = VGG16(weights=None, include_top=False, input_shape=(IMG, IMG, 3))
    inp = layers.Input((IMG, IMG, 3))
    x = layers.Lambda(preprocess_input)(inp)
    x = base(x, training=False)
    gap = layers.GlobalAveragePooling2D()
    x = gap(x)
    x = layers.Dropout(0.5)(x)
    head = layers.Dense(1, activation="sigmoid")
    out = head(x)
    model = keras.Model(inp, out)
    model.load_weights(path)
    conv_model = keras.Model(
        base.input,
        [base.get_layer("block5_conv3").output, base.output],
    )
    return model, conv_model, gap, head


def to_rgb_white_bg(img_pil: Image.Image) -> Image.Image:
    """Convert any image (including transparent PNGs) to RGB on a white background."""
    if img_pil.mode in ("RGBA", "LA") or (
        img_pil.mode == "P" and "transparency" in img_pil.info
    ):
        rgba = img_pil.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img_pil.convert("RGB")


def predict_with_gradcam(model, conv_model, gap, head, img_pil: Image.Image):
    """Run VGG16 prediction and compute Grad-CAM attribution map."""
    resample = getattr(Image, "Resampling", Image).BILINEAR
    resized = img_pil.resize((IMG, IMG), resample)
    arr = np.array(resized, dtype="float32")
    batch = arr[None, ...]

    prob = float(model.predict(batch, verbose=0)[0][0])

    x_pre = preprocess_input(batch.copy())
    x_tf = tf.convert_to_tensor(x_pre, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        feat, conv_out = conv_model(x_tf, training=False)
        gap_out = gap(conv_out)
        try:
            logit = tf.matmul(gap_out, tf.convert_to_tensor(head.kernel)) + tf.convert_to_tensor(head.bias)
            target = logit if prob >= 0.5 else -logit
        except Exception:
            p_out = head(gap_out)
            target = p_out if prob >= 0.5 else (1.0 - p_out)

    grads = tape.gradient(target, feat)
    if grads is None:
        return prob, np.zeros((14, 14), dtype=np.float32)

    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * feat, axis=-1))[0].numpy()
    max_val = float(cam.max())
    if max_val > 1e-8:
        cam = cam / max_val
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return prob, cam.astype(np.float32)


def render_gradcam_overlay(
    img_pil: Image.Image,
    cam: np.ndarray,
    max_alpha: float = 0.62,
) -> Image.Image:
    """Blend a transparent clinical heatmap over the original drawing."""
    base_rgba = img_pil.convert("RGBA")
    w, h = base_rgba.size
    resample = getattr(Image, "Resampling", Image).BILINEAR

    cam_img = Image.fromarray(np.clip(cam * 255.0, 0, 255).astype(np.uint8), mode="L")
    cam_resized = np.array(
        cam_img.resize((w, h), resample),
        dtype=np.float32,
    ) / 255.0

    # Smooth 3-stop clinical palette: Teal (0.0) -> Amber (0.5) -> Red (1.0)
    stops = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    r_ch = np.interp(cam_resized, stops, [20.0, 245.0, 239.0])
    g_ch = np.interp(cam_resized, stops, [184.0, 158.0, 68.0])
    b_ch = np.interp(cam_resized, stops, [166.0, 11.0, 68.0])

    # Keep low-activation paper background clean and readable
    alpha = np.clip((cam_resized - 0.16) / 0.84, 0.0, 1.0) ** 1.15
    a_ch = alpha * (max_alpha * 255.0)

    heatmap_rgba = np.stack([r_ch, g_ch, b_ch, a_ch], axis=-1).astype(np.uint8)
    overlay = Image.fromarray(heatmap_rgba, mode="RGBA")
    return Image.alpha_composite(base_rgba, overlay).convert("RGB")


def interpret_probability(prob: float) -> dict:
    """Map raw Parkinson probability into plain-language clinical guidance."""
    pct = prob * 100.0
    if prob >= INCONCLUSIVE_HIGH:
        risk = "High Likelihood" if prob >= HIGH_RISK_THRESHOLD else "Moderate-High Likelihood"
        return {
            "css": "warn",
            "badge": "⚠️ Parkinson-like Pattern Detected",
            "risk": risk,
            "chance_text": f"{pct:.1f}%",
            "meaning": (
                "The AI found stroke irregularities—such as tremor oscillations, uneven spacing, "
                "or sudden directional shifts—that closely match Parkinson's drawings in its training data."
            ),
            "cam_caption": (
                "Highlighted red & orange areas show the exact stroke regions where Parkinson-like cues were detected."
            ),
            "next_step": "Because this is an AI screening aid (81.7% test accuracy), consider sharing results with a neurologist for proper clinical evaluation.",
        }
    if prob >= INCONCLUSIVE_LOW:
        return {
            "css": "unsure",
            "badge": "❓ Borderline / Inconclusive",
            "risk": "Uncertain (Near 50/50)",
            "chance_text": f"{pct:.1f}%",
            "meaning": (
                "The drawing sits near the middle of the scale, showing a mix of steady and uneven strokes. "
                "The model cannot make a confident call from this image alone."
            ),
            "cam_caption": (
                "Highlighted regions show where the model saw mild irregularities while weighing both classes."
            ),
            "next_step": "Try uploading a sharper, well-lit photo showing the complete spiral or wave on plain paper.",
        }
    return {
        "css": "ok",
        "badge": "✅ Healthy-like Pattern",
        "risk": "Low Likelihood",
        "chance_text": f"{pct:.1f}%",
        "meaning": (
            "The pen strokes appear smooth, continuous, and evenly spaced—consistent with healthy control drawings."
        ),
        "cam_caption": (
            "Highlighted areas show the smooth, steady stroke segments that most strongly supported the Healthy result."
        ),
        "next_step": "No Parkinson-like drawing pattern was flagged, though persistent physical symptoms should always be checked by a doctor.",
    }


def render_hero() -> None:
    """Render top hero header and 3-step user guide."""
    st.markdown(
        """
<div class="hero-card">
  <div class="hero-badge">🌀 AI Screening Aid · VGG16 + Grad-CAM Explainability</div>
  <h1 class="hero-title">Parkinson's Disease Drawing Screening</h1>
  <p class="hero-sub">
    Upload one or multiple hand-drawn <b>spiral</b> or <b>wave</b> images. Our AI estimates the chance of
    Parkinson-like tremor patterns and highlights <b>exactly where</b> on the drawing those cues appear.
  </p>
</div>
<div class="step-strip">
  <div class="step-pill">
    <span class="step-num">1</span>
    <div class="step-text">
      <strong>Upload 1 to 3+ Drawings</strong>
      <span>Select multiple spiral or wave photos at once</span>
    </div>
  </div>
  <div class="step-pill">
    <span class="step-num">2</span>
    <div class="step-text">
      <strong>See Exact Chance %</strong>
      <span>Clear risk score &amp; plain-English explanation</span>
    </div>
  </div>
  <div class="step-pill">
    <span class="step-num">3</span>
    <div class="step-text">
      <strong>Spot Where It Detected</strong>
      <span>Visual heatmap pinpoints key pen strokes</span>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_gauge_svg(prob: float, css_cls: str) -> str:
    """Build a crisp circular progress ring SVG for Parkinson likelihood."""
    pct = max(0.0, min(100.0, prob * 100.0))
    radius = 28
    circumference = 2.0 * 3.14159265 * radius
    dash = (pct / 100.0) * circumference
    color_map = {"warn": "#D97706", "unsure": "#4F46E5", "ok": "#0F766E"}
    stroke_color = color_map.get(css_cls, "#0F766E")
    return f"""
<svg width="72" height="72" viewBox="0 0 72 72" aria-label="{pct:.1f}% chance">
  <circle cx="36" cy="36" r="{radius}" fill="none" stroke="#E2E8F0" stroke-width="7"/>
  <circle cx="36" cy="36" r="{radius}" fill="none" stroke="{stroke_color}" stroke-width="7"
          stroke-linecap="round" stroke-dasharray="{dash:.2f} {circumference:.2f}"
          transform="rotate(-90 36 36)"/>
  <text x="36" y="40" text-anchor="middle" font-family="JetBrains Mono, monospace"
        font-size="13" font-weight="800" fill="#0F172A">{pct:.0f}%</text>
</svg>"""


def render_result_card(
    index: int,
    filename: str,
    img_pil: Image.Image,
    overlay_pil: Image.Image,
    prob: float,
    compact: bool = False,
) -> None:
    """Render an individual, easy-to-read diagnostic card for one drawing."""
    info = interpret_probability(prob)
    safe_name = html.escape(filename)
    parkinson_pct = prob * 100.0
    healthy_pct = (1.0 - prob) * 100.0
    gauge_svg = build_gauge_svg(prob, info["css"])

    details_html = f"""
<div class="chance-card {info['css']}">
  <div class="chance-top">
    <div style="display:flex;align-items:center;gap:0.85rem;">
      {gauge_svg}
      <div>
        <div class="chance-label">Parkinson's Pattern Chance</div>
        <div class="chance-big {info['css']}">{info['chance_text']}</div>
      </div>
    </div>
    <span class="risk-tag">{info['risk']}</span>
  </div>
  <div class="prob-row">
    <span>Parkinson-like cues</span>
    <span>{parkinson_pct:.1f}%</span>
  </div>
  <div class="prob-track">
    <div class="prob-fill" style="width:{parkinson_pct:.1f}%;background:#D97706;"></div>
  </div>
  <div class="prob-row">
    <span>Healthy / steady cues</span>
    <span>{healthy_pct:.1f}%</span>
  </div>
  <div class="prob-track">
    <div class="prob-fill" style="width:{healthy_pct:.1f}%;background:#0F766E;"></div>
  </div>
</div>
<div class="explain-box">
  <b>What this means:</b> {info['meaning']}<br><br>
  <b>Suggested next step:</b> {info['next_step']}
</div>
"""

    legend_html = f"""
<div class="cam-legend">
  <span>Low focus</span>
  <div class="legend-bar"></div>
  <span>Strongest cue</span>
</div>
<div style="font-size:0.84rem;color:#475569;margin-top:0.35rem;line-height:1.45;">
  {info['cam_caption']}
</div>
"""

    with st.container(border=True):
        st.markdown(
            f"""
<div class="result-header">
  <span class="file-chip">#{index} · {safe_name}</span>
  <span class="status-badge {info['css']}">{info['badge']}</span>
</div>
""",
            unsafe_allow_html=True,
        )

        if compact:
            c_orig, c_cam = st.columns(2, gap="small")
            with c_orig:
                st.caption("Original Drawing")
                st.image(img_pil, use_container_width=True)
            with c_cam:
                st.caption("Where AI Focused (Grad-CAM)")
                st.image(overlay_pil, use_container_width=True)
            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
            st.markdown(details_html, unsafe_allow_html=True)
        else:
            col_imgs, col_details = st.columns([1.15, 1.0], gap="large")
            with col_imgs:
                c_orig, c_cam = st.columns(2, gap="small")
                with c_orig:
                    st.caption("Original Drawing")
                    st.image(img_pil, use_container_width=True)
                with c_cam:
                    st.caption("Where AI Focused (Grad-CAM)")
                    st.image(overlay_pil, use_container_width=True)
                st.markdown(legend_html, unsafe_allow_html=True)
            with col_details:
                st.markdown(details_html, unsafe_allow_html=True)


def render_screening_tab(model, conv_model, gap, head) -> None:
    """Render multi-image upload screening tab."""
    with st.container(border=True):
        h_col, opt_col = st.columns([2.2, 1.0], gap="medium")
        with h_col:
            st.markdown(
                '<div class="section-heading">Upload Spiral or Wave Drawings</div>'
                '<div class="section-caption">You can select 1, 2, 3, or more PNG/JPG images together in one upload.</div>',
                unsafe_allow_html=True,
            )
        with opt_col:
            heatmap_strength = st.slider(
                "Heatmap overlay intensity",
                min_value=0.25,
                max_value=0.85,
                value=0.62,
                step=0.05,
                help="Adjust how strongly the Grad-CAM focus highlights appear over your drawing.",
            )

        uploaded_files = st.file_uploader(
            "Choose drawing images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    if not uploaded_files:
        st.markdown(
            """
<div class="tips-grid">
  <div class="tip-card">
    <div class="tip-icon">📄</div>
    <div class="tip-title">Use a Clean Page</div>
    <div class="tip-desc">Plain white paper with dark pen ink gives the clearest stroke detection.</div>
  </div>
  <div class="tip-card">
    <div class="tip-icon">💡</div>
    <div class="tip-title">Bright, Even Lighting</div>
    <div class="tip-desc">Avoid heavy shadows across the spiral or wave so false edges aren't flagged.</div>
  </div>
  <div class="tip-card">
    <div class="tip-icon">📚</div>
    <div class="tip-title">Compare 2–3 Drawings</div>
    <div class="tip-desc">Upload multiple spirals or waves at once to see individual chances &amp; heatmaps side by side.</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    if len(uploaded_files) > 6:
        st.info(f"Analysing {len(uploaded_files)} drawings. Larger batches may take a few extra seconds.")

    results = []
    progress = st.progress(0, text="Analysing uploaded drawings...")
    total = len(uploaded_files)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for idx, f in enumerate(uploaded_files, start=1):
        progress.progress(idx / total, text=f"Analysing drawing {idx} of {total}: {f.name}")
        try:
            img_pil = to_rgb_white_bg(Image.open(f))
            img_pil.thumbnail((900, 900), resample)
        except Exception:
            st.warning(f"Could not read `{f.name}` as a valid image. Please upload a PNG or JPG file.")
            continue
        prob, cam = predict_with_gradcam(model, conv_model, gap, head, img_pil)
        overlay_pil = render_gradcam_overlay(img_pil, cam, max_alpha=heatmap_strength)
        results.append((f.name, img_pil, overlay_pil, prob))
    progress.empty()

    if not results:
        return

    if len(results) > 1:
        n_parkinson = sum(1 for _, _, _, p in results if p >= INCONCLUSIVE_HIGH)
        n_unsure = sum(1 for _, _, _, p in results if INCONCLUSIVE_LOW <= p < INCONCLUSIVE_HIGH)
        n_healthy = sum(1 for _, _, _, p in results if p < INCONCLUSIVE_LOW)
        st.markdown(
            f"""
<div class="batch-bar">
  <div class="batch-title">Batch Summary · {len(results)} Drawings Analysed</div>
  <div class="batch-pills">
    <span class="mini-pill warn">{n_parkinson} Parkinson-like</span>
    <span class="mini-pill unsure">{n_unsure} Borderline</span>
    <span class="mini-pill ok">{n_healthy} Healthy-like</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        f_col, v_col = st.columns([2.2, 1.0], gap="medium")
        with f_col:
            filter_choice = st.radio(
                "Filter results",
                options=[
                    f"All ({len(results)})",
                    f"Parkinson-like ({n_parkinson})",
                    f"Borderline ({n_unsure})",
                    f"Healthy-like ({n_healthy})",
                ],
                horizontal=True,
                label_visibility="collapsed",
            )
        with v_col:
            view_mode = st.radio(
                "Layout",
                options=["2-Column Grid", "Full-Width Rows"],
                horizontal=True,
                label_visibility="collapsed",
            )

        indexed_results = list(enumerate(results, start=1))
        if filter_choice.startswith("Parkinson-like"):
            filtered = [(i, item) for i, item in indexed_results if item[3] >= INCONCLUSIVE_HIGH]
        elif filter_choice.startswith("Borderline"):
            filtered = [
                (i, item)
                for i, item in indexed_results
                if INCONCLUSIVE_LOW <= item[3] < INCONCLUSIVE_HIGH
            ]
        elif filter_choice.startswith("Healthy-like"):
            filtered = [(i, item) for i, item in indexed_results if item[3] < INCONCLUSIVE_LOW]
        else:
            filtered = indexed_results

        if not filtered:
            st.info("No drawings match this filter.")
            return

        if view_mode == "2-Column Grid" and len(filtered) > 1:
            for row_start in range(0, len(filtered), 2):
                cols = st.columns(2, gap="medium")
                for col, (orig_idx, (fname, img_pil, overlay_pil, prob)) in zip(
                    cols, filtered[row_start : row_start + 2]
                ):
                    with col:
                        render_result_card(
                            orig_idx, fname, img_pil, overlay_pil, prob, compact=True
                        )
        else:
            for orig_idx, (fname, img_pil, overlay_pil, prob) in filtered:
                render_result_card(orig_idx, fname, img_pil, overlay_pil, prob, compact=False)
    else:
        fname, img_pil, overlay_pil, prob = results[0]
        render_result_card(1, fname, img_pil, overlay_pil, prob, compact=False)


def render_benchmarks_tab() -> None:
    """Render model comparison metrics and confusion matrix from Parkinson.ipynb."""
    st.markdown(
        """
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-val">81.7%</div>
    <div class="kpi-lbl">Test accuracy on 60 unseen drawings (95% CI: 70%–91.7%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">86.7%</div>
    <div class="kpi-lbl">Parkinson's recall / sensitivity (26 of 30 cases caught)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">0.914</div>
    <div class="kpi-lbl">ROC-AUC score (1.000 is perfect separation)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">204</div>
    <div class="kpi-lbl">Unique drawings (115 train · 29 validation · 60 test)</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_list, col_cm = st.columns([1.45, 1.0], gap="large")

    with col_list:
        st.markdown(
            '<div class="section-heading">Model Comparison on Unseen Test Set (60 Images)</div>'
            '<div class="section-caption">Why VGG16 was chosen over Custom CNN, ResNet50, and Ensemble.</div>',
            unsafe_allow_html=True,
        )
        for m in MODEL_BENCHMARKS:
            active_cls = "active" if m["highlight"] else ""
            bar_color = "#0F766E" if m["highlight"] else "#64748B"
            spec_str = f"{m['specificity']:.1f}%" if m["specificity"] is not None else "—"
            st.markdown(
                f"""
<div class="bench-row {active_cls}">
  <div class="bench-head">
    <span class="bench-name">{m['name']}</span>
    <span class="bench-tag">{m['tag']}</span>
  </div>
  <div class="prob-track" style="height:10px;">
    <div class="prob-fill" style="width:{m['test_acc']}%;background:{bar_color};"></div>
  </div>
  <div class="bench-metrics">
    <span>Accuracy: <b>{m['test_acc']:.1f}%</b></span>
    <span>Recall: <b>{m['recall']:.1f}%</b></span>
    <span>F1: <b>{m['f1']:.1f}%</b></span>
    <span>Specificity: <b>{spec_str}</b></span>
    <span>AUC: <b>{m['auc']:.3f}</b></span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with col_cm:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">VGG16 Test Confusion Matrix</div>'
                '<div class="section-caption">Breakdown across 30 Healthy and 30 Parkinson test drawings.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
<div class="cm-grid">
  <div class="cm-cell good">
    <div class="cm-num">23</div>
    <div class="cm-cap">True Healthy<br>(Correct)</div>
  </div>
  <div class="cm-cell bad">
    <div class="cm-num">7</div>
    <div class="cm-cap">False Positive<br>(Flagged as PD)</div>
  </div>
  <div class="cm-cell bad">
    <div class="cm-num">4</div>
    <div class="cm-cap">False Negative<br>(Missed PD)</div>
  </div>
  <div class="cm-cell good">
    <div class="cm-num">26</div>
    <div class="cm-cap">True Parkinson<br>(Correct)</div>
  </div>
</div>
<div class="explain-box" style="margin-top:1rem;">
  <b>Key Takeaway:</b> Fine-tuned <b>VGG16</b> achieved the highest sensitivity (<b>86.7%</b>) and overall
  F1 score (<b>82.5%</b>), missing only 4 out of 30 Parkinson cases while training from scratch with a small
  CNN collapsed to 50% accuracy.
</div>
""",
                unsafe_allow_html=True,
            )


def render_guide_tab() -> None:
    """Render plain-language explanation of how the system and Grad-CAM work."""
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">How the AI Analyses Drawings</div>'
                '<div class="section-caption">From raw photo to tremor heatmap in 3 stages.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
1. **Standardization (`224 × 224 RGB`):** Every uploaded spiral or wave photo is converted to 3-channel RGB, resized cleanly, and normalized with VGG16 preprocessing.
2. **Feature Extraction (`VGG16`):** Pre-trained ImageNet convolutional filters—fine-tuned on the last 20 layers using the Kaggle Parkinson's Drawings dataset—scan the pen lines for tremor oscillations and irregular spacing.
3. **Where It Detected (`Grad-CAM`):** Gradients from the final prediction flow back to the last convolutional layer (`block5_conv3`) to highlight the exact regions of the drawing that drove the probability score.
"""
            )
    with c2:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">Important Limitations to Know</div>'
                '<div class="section-caption">Transparency &amp; responsible AI notes.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
- **Small Research Dataset:** Trained and validated on **204 unique drawings** (115 train, 29 validation, 60 test). Different pens, lined paper, or camera angles may lower reliability.
- **Confidence Range:** Test accuracy is **81.7%** (95% confidence interval: **70.0% to 91.7%**). Scores between **35% and 65%** are marked **Borderline / Inconclusive**.
- **Domain Scope:** Only spiral and wave drawings are valid inputs; unrelated photos will produce meaningless outputs.
- **Not a Medical Diagnosis:** Built strictly for research, screening exploration, and education. Always consult a qualified neurologist for medical decisions.
"""
            )


def main() -> None:
    inject_styles()
    model, conv_model, gap, head = load_model()
    render_hero()

    tab_screen, tab_bench, tab_guide = st.tabs(
        ["🔬 Screen Drawings", "📊 Model & Accuracy", "ℹ️ How It Works"]
    )
    with tab_screen:
        render_screening_tab(model, conv_model, gap, head)
    with tab_bench:
        render_benchmarks_tab()
    with tab_guide:
        render_guide_tab()

    st.markdown(
        """
<div class="medical-banner">
  <b>Medical Disclaimer:</b> This application is a deep learning research &amp; educational project
  (<i>Project Cheyali</i>) and is not a certified medical device or clinical diagnosis.
</div>
<div class="app-footer">
  Project Cheyali · Dataset: Kaggle Parkinson's Drawings (204 unique images) · Model: Fine-tuned VGG16 + Grad-CAM
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
