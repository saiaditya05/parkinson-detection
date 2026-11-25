import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from huggingface_hub import hf_hub_download

IMG = 224

st.set_page_config(page_title="Parkinson's Drawing Screening", page_icon="🌀", layout="wide")

st.markdown(
    """
<style>
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --brand:#4338ca; --brand2:#0ea5e9;
        --ok:#0f766e; --ok-bg:#ccfbf1; --warn:#b45309; --warn-bg:#fef3c7; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; height:0; }
.block-container { padding-top:1.2rem; max-width:1100px; }
.hero { background:linear-gradient(120deg,#312e81 0%,#4338ca 45%,#0ea5e9 100%);
        color:#fff; padding:2.2rem 2.2rem 2rem; border-radius:20px; margin-bottom:1.4rem; }
.hero h1 { font-size:2.1rem; margin:0 0 .35rem; font-weight:800; letter-spacing:-.02em; color:#fff; }
.hero p { margin:0; font-size:1.02rem; opacity:.92; max-width:640px; }
.pill { display:inline-block; background:rgba(255,255,255,.18); padding:.2rem .7rem; border-radius:999px;
        font-size:.78rem; margin:0 .4rem .8rem 0; }
.card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.3rem 1.4rem;
        box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05); }
.card h3 { margin:0 0 .6rem; font-size:1.05rem; color:var(--ink); }
.empty { text-align:center; color:var(--muted); padding:3.2rem 1rem; }
.empty .big { font-size:2.6rem; }
.badge { display:inline-block; font-weight:700; font-size:1.35rem; padding:.45rem 1rem; border-radius:12px; }
.badge.ok { background:var(--ok-bg); color:var(--ok); }
.badge.warn { background:var(--warn-bg); color:var(--warn); }
.bar-row { margin:.9rem 0 .2rem; font-size:.92rem; color:var(--ink); display:flex; justify-content:space-between; }
.bar { height:12px; background:#f1f5f9; border-radius:999px; overflow:hidden; }
.bar > div { height:100%; border-radius:999px; }
.note { margin-top:1rem; padding:.7rem .9rem; background:#f8fafc; border-left:4px solid var(--brand);
        border-radius:8px; color:var(--muted); font-size:.88rem; }
.tile { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
.tile .v { font-size:1.6rem; font-weight:800; color:var(--brand); letter-spacing:-.02em; }
.tile .l { color:var(--muted); font-size:.85rem; margin-top:.1rem; }
.foot { text-align:center; color:var(--muted); font-size:.8rem; margin:2rem 0 .5rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = hf_hub_download("saiaditya05/cheyali-vgg16", "vgg16.weights.h5")
    base = VGG16(weights=None, include_top=False, input_shape=(IMG, IMG, 3))
    inp = layers.Input((IMG, IMG, 3))
    x = layers.Lambda(preprocess_input)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    m = keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))
    m.load_weights(path)
    return m


model = load_model()

st.markdown(
    """
<div class="hero">
  <span class="pill">Deep learning</span><span class="pill">VGG16 transfer learning</span><span class="pill">Research project</span>
  <h1>Parkinson's Drawing Screening</h1>
  <p>Upload a hand-drawn spiral or wave. The model looks for tremor patterns in the pen strokes and
  estimates whether the drawing looks healthy or Parkinson-like.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card"><h3>1. Upload a drawing</h3>', unsafe_allow_html=True)
    f = st.file_uploader("Spiral or wave drawing (PNG / JPG)", type=["png", "jpg", "jpeg"],
                         label_visibility="collapsed")
    if f:
        img = Image.open(f).convert("RGB")
        st.image(img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><h3>2. Result</h3>', unsafe_allow_html=True)
    if not f:
        st.markdown(
            '<div class="empty"><div class="big">🌀</div>'
            "Upload a spiral or wave drawing<br>to see the prediction here.</div>",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analysing drawing..."):
            x = np.array(img.resize((IMG, IMG)), dtype="float32")[None]
            p = float(model.predict(x, verbose=0)[0][0])
        is_pd = p >= 0.5
        label = "Parkinson-like pattern" if is_pd else "Healthy-like pattern"
        cls = "warn" if is_pd else "ok"
        conf = p if is_pd else 1 - p
        st.markdown(
            f"""
<span class="badge {cls}">{label}</span>
<div style="color:#64748b;margin-top:.5rem;font-size:.92rem">Model confidence: <b>{conf*100:.0f}%</b></div>
<div class="bar-row"><span>Parkinson-like</span><b>{p*100:.1f}%</b></div>
<div class="bar"><div style="width:{p*100:.1f}%;background:#b45309"></div></div>
<div class="bar-row"><span>Healthy-like</span><b>{(1-p)*100:.1f}%</b></div>
<div class="bar"><div style="width:{(1-p)*100:.1f}%;background:#0f766e"></div></div>
""",
            unsafe_allow_html=True,
        )
        if 0.35 <= p <= 0.65:
            st.markdown('<div class="note">The model is unsure about this drawing (close to 50/50). '
                        "Treat this result as inconclusive.</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="note">This is a screening aid for a learning project, not a diagnosis. '
                        "Please consult a neurologist for any medical concern.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
t1, t2, t3, t4 = st.columns(4)
for col, v, l in [
    (t1, "81.7%", "Test accuracy (60 images)"),
    (t2, "0.91", "AUC on test set"),
    (t3, "204", "Spiral + wave drawings"),
    (t4, "VGG16", "Fine-tuned from ImageNet"),
]:
    col.markdown(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("How it works"):
    st.write(
        "Drawings are resized to 224x224 and passed through a VGG16 network pre-trained on ImageNet, "
        "then fine-tuned on hand-drawn spirals and waves from healthy people and people with Parkinson's disease. "
        "The final layer outputs the probability that the drawing is Parkinson-like."
    )
with st.expander("Limitations"):
    st.write(
        "- Trained on only 204 drawings from 55 people, so results may not generalise to new people, pens or paper.\n"
        "- Test accuracy is 81.7% (95% confidence interval roughly 70-92%) on 60 test images.\n"
        "- Photos that are not spiral or wave drawings give meaningless results.\n"
        "- This is a research and learning project, not a medical device."
    )

st.markdown('<div class="foot">Project Cheyali · Data: Kaggle Parkinson\'s Drawings · Model: VGG16</div>',
            unsafe_allow_html=True)
