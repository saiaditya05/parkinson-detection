import numpy as np, streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from huggingface_hub import hf_hub_download
IMG = 224
@st.cache_resource
def load_model():
    path = hf_hub_download("saiaditya05/cheyali-vgg16", "vgg16.weights.h5")
    base = VGG16(weights=None, include_top=False, input_shape=(IMG,IMG,3))
    inp = layers.Input((IMG,IMG,3)); x = layers.Lambda(preprocess_input)(inp)
    x = base(x, training=False); x = layers.GlobalAveragePooling2D()(x); x = layers.Dropout(0.5)(x)
    m = keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))
    m.load_weights(path)
    return m
model = load_model()
st.title("Parkinson's Disease Detection")
st.caption("Research/learning project, not a medical diagnosis. Upload a spiral or wave drawing.")
f = st.file_uploader("Upload drawing", type=["png","jpg","jpeg"])
if f:
    img = Image.open(f).convert("RGB"); st.image(img, width=300)
    x = np.array(img.resize((IMG,IMG)), dtype="float32")[None]
    p = float(model.predict(x, verbose=0)[0][0])
    st.subheader("Parkinson" if p >= 0.5 else "Healthy")
    st.write(f"P(parkinson) = {p:.2f}, P(healthy) = {1-p:.2f}")
