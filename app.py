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
html, body, [class*="css"] { font-size:17px; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.5rem; max-width:1150px; }

.hero { background:linear-gradient(120deg,#1e1b4b 0%,#3730a3 50%,#0369a1 100%);
        color:#fff; padding:2.4rem 2.6rem; border-radius:22px; margin-bottom:1.6rem; }
.hero .tag { font-size:.8rem; letter-spacing:.12em; text-transform:uppercase; opacity:.8; margin-bottom:.5rem; }
.hero h1 { font-size:2.5rem; line-height:1.15; margin:0 0 .6rem; font-weight:800; color:#fff; padding:0; }
.hero p { margin:0; font-size:1.1rem; line-height:1.55; opacity:.95; max-width:720px; }

.steps { display:flex; gap:.8rem; margin-bottom:1.4rem; flex-wrap:wrap; }
.step { flex:1; min-width:200px; background:#f1f5f9; border-radius:14px; padding:.9rem 1.1rem; color:#0f172a; }
.step b { display:inline-block; width:1.7rem; height:1.7rem; line-height:1.7rem; text-align:center;
          background:#3730a3; color:#fff; border-radius:50%; margin-right:.55rem; font-size:.95rem; }

.panel-title { font-size:1.35rem; font-weight:700; color:#0f172a; margin:0 0 .2rem; }
.panel-sub { color:#64748b; font-size:.98rem; margin-bottom:.6rem; }

.verdict { display:flex; align-items:center; gap:1rem; padding:1.2rem 1.3rem; border-radius:16px; margin:.3rem 0 1.1rem; }
.verdict.ok { background:#dcfce7; color:#14532d; }
.verdict.warn { background:#fef3c7; color:#78350f; }
.verdict.unsure { background:#e0e7ff; color:#312e81; }
.verdict .icon { font-size:2.6rem; line-height:1; }
.verdict .vlabel { font-size:1.7rem; font-weight:800; line-height:1.15; }
.verdict .vsub { font-size:1.02rem; opacity:.85; margin-top:.15rem; }

.bar-row { display:flex; justify-content:space-between; font-size:1.05rem; color:#0f172a; margin:.8rem 0 .3rem; font-weight:600; }
.bar { height:16px; background:#e2e8f0; border-radius:999px; overflow:hidden; }
.bar > div { height:100%; border-radius:999px; }

.meaning { margin-top:1.2rem; font-size:1.02rem; line-height:1.55; color:#334155; }
.note { margin-top:1rem; padding:.85rem 1rem; background:#f8fafc; border-left:5px solid #3730a3;
        border-radius:8px; color:#475569; font-size:.95rem; line-height:1.5; }

.empty { text-align:center; color:#64748b; padding:3.5rem 1rem 3rem; font-size:1.05rem; line-height:1.6; }
.empty .big { font-size:3.2rem; margin-bottom:.4rem; }

.tile { background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:1.1rem 1.2rem; min-height:8.2rem; }
.tile .v { font-size:2rem; font-weight:800; color:#3730a3; letter-spacing:-.02em; line-height:1.1; }
.tile .l { color:#64748b; font-size:.95rem; margin-top:.25rem; }
.foot { text-align:center; color:#94a3b8; font-size:.85rem; margin:2rem 0 .5rem; }
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
  <div class="tag">Deep learning research project</div>
  <h1>Can a hand-drawn spiral reveal Parkinson's?</h1>
  <p>Upload a photo or scan of a spiral or wave drawing. The model checks the pen strokes for tremor-like
  patterns and tells you whether the drawing looks <b>healthy</b> or <b>Parkinson-like</b>.</p>
</div>
<div class="steps">
  <div class="step"><b>1</b>Upload a spiral or wave drawing</div>
  <div class="step"><b>2</b>The AI analyses the strokes</div>
  <div class="step"><b>3</b>See the result and confidence</div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">1. Your drawing</div>'
                    '<div class="panel-sub">PNG or JPG of a spiral or wave drawn on paper</div>',
                    unsafe_allow_html=True)
        f = st.file_uploader("Upload drawing", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if f:
            img = Image.open(f).convert("RGB")
            st.image(img, use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">2. Result</div>'
                    '<div class="panel-sub">What the model thinks of this drawing</div>',
                    unsafe_allow_html=True)
        if not f:
            st.markdown(
                '<div class="empty"><div class="big">🌀</div>'
                "No drawing yet.<br>Upload one on the left and the result will appear here.</div>",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analysing drawing..."):
                x = np.array(img.resize((IMG, IMG)), dtype="float32")[None]
                p = float(model.predict(x, verbose=0)[0][0])
            conf = max(p, 1 - p)
            if 0.35 <= p <= 0.65:
                cls, icon, label = "unsure", "❓", "Inconclusive"
                sub = "The model is not sure about this drawing."
                meaning = ("The score is close to 50/50, so the model cannot tell. "
                           "Try a clearer, well-lit photo of the full drawing.")
            elif p > 0.65:
                cls, icon, label = "warn", "⚠️", "Parkinson-like pattern"
                sub = f"Model confidence: {conf*100:.0f}%"
                meaning = ("The strokes show features the model links to Parkinson's drawings, "
                           "such as shaky or irregular lines.")
            else:
                cls, icon, label = "ok", "✅", "Healthy-like pattern"
                sub = f"Model confidence: {conf*100:.0f}%"
                meaning = ("The strokes look smooth and steady, like the healthy drawings "
                           "the model was trained on.")
            st.markdown(
                f"""
<div class="verdict {cls}"><div class="icon">{icon}</div>
  <div><div class="vlabel">{label}</div><div class="vsub">{sub}</div></div></div>
<div class="bar-row"><span>Parkinson-like</span><span>{p*100:.1f}%</span></div>
<div class="bar"><div style="width:{p*100:.1f}%;background:#d97706"></div></div>
<div class="bar-row"><span>Healthy-like</span><span>{(1-p)*100:.1f}%</span></div>
<div class="bar"><div style="width:{(1-p)*100:.1f}%;background:#0d9488"></div></div>
<div class="meaning"><b>What this means:</b> {meaning}</div>
<div class="note">This is a learning project and not a medical diagnosis.
Please consult a neurologist for any health concern.</div>
""",
                unsafe_allow_html=True,
            )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="panel-title">About the model</div>', unsafe_allow_html=True)
t1, t2, t3, t4 = st.columns(4)
for col, v, l in [
    (t1, "81.7%", "Accuracy on 60 unseen test drawings"),
    (t2, "0.91", "AUC (1.0 is perfect)"),
    (t3, "204", "Training drawings (spiral + wave)"),
    (t4, "VGG16", "Pre-trained network, fine-tuned"),
]:
    col.markdown(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("How it works"):
    st.write(
        "The drawing is resized to 224x224 pixels and passed through VGG16, an image network pre-trained on "
        "ImageNet. It was fine-tuned on hand-drawn spirals and waves from healthy people and people with "
        "Parkinson's disease. The last layer gives the probability that the drawing is Parkinson-like."
    )
with st.expander("Limitations"):
    st.write(
        "- Trained on only 204 drawings from 55 people, so it may not work well for new people, pens or paper.\n"
        "- Test accuracy is 81.7% (95% confidence interval about 70% to 92%) on 60 images.\n"
        "- Photos that are not spiral or wave drawings give meaningless results.\n"
        "- This is a research and learning project, not a medical device."
    )

st.markdown('<div class="foot">Project Cheyali · Data: Kaggle Parkinson\'s Drawings · Model: VGG16</div>',
            unsafe_allow_html=True)
