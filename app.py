
import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="MediX", page_icon="🌿")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("medix_plant_disease.keras")
    with open("class_names.json", "r") as f:
        classes = json.load(f)
    return model, classes

st.title("🌿 MediX")
st.subheader("Plant Disease Detection")
st.write("Upload a clear image of a plant leaf.")

try:
    model, classes = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if st.button("Detect Disease"):
        shape = model.input_shape
        resized = image.resize((shape[2], shape[1]))
        array = np.array(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        with st.spinner("Analyzing leaf..."):
            result = model.predict(array, verbose=0)[0]

        index = int(np.argmax(result))
        label = classes[index] if isinstance(classes, list) else list(classes.values())[index]

        st.markdown("### 🌱 Prediction")
        st.write(f"**Class:** {label}")
        st.write(f"**Confidence:** {result[index] * 100:.2f}%")
        st.progress(float(np.clip(result[index], 0, 1)))
        st.caption("AI predictions can be wrong. Confirm treatment with an agricultural expert.")
