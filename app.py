 
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# --------------------------------------------------
# MediX: Plant Disease Detection
# --------------------------------------------------

st.set_page_config(
    page_title="MediX | Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

MODEL_PATH = Path("medix_plant_disease.keras")
CLASS_PATH = Path("class_names.json")


# Load the existing trained model only once
@st.cache_resource
def load_medix_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError(
            "class_names.json must contain a list of class names."
        )

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError(
            "This app expects a single-output classification model."
        )

    if output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model outputs {output_shape[-1]} classes, "
            f"but class_names.json contains {len(class_names)}."
        )

    return model, class_names


def prepare_image(image, model):
    """Resize an RGB image to the model's input dimensions."""
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        raise ValueError("A single-input model is required.")

    height, width, channels = input_shape[1:4]

    if height is None or width is None:
        raise ValueError(
            "The model's image dimensions are not fixed."
        )

    if channels != 3:
        raise ValueError(
            "This app expects a model that accepts RGB images."
        )

    # Preserve EXIF orientation and convert to RGB.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(
        (int(width), int(height)),
        Image.Resampling.LANCZOS,
    )

    # Keep pixel values in the 0-255 range.
    # Change this ONLY after verifying training preprocessing.
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def explain_class(class_name):
    """Provide general information for common PlantVillage classes."""
    name = class_name.lower()

    if "healthy" in name:
        return (
            "The model predicts a healthy leaf. Continue regular "
            "monitoring and maintain suitable watering and nutrition."
        )

    if "black_rot" in name:
        return (
            "Black rot is a possible prediction. Inspect for spreading "
            "lesions, remove severely affected plant material where "
            "appropriate, and avoid overhead watering. Confirm the "
            "plant species and diagnosis before treatment."
        )

    if "rust" in name:
        return (
            "Rust is a possible prediction. Check for rust-coloured "
            "spots or pustules, improve airflow where appropriate, "
            "and seek local agricultural advice if it spreads."
        )

    if "powdery_mildew" in name:
        return (
            "Powdery mildew is a possible prediction. Look for a "
            "white powdery coating, improve airflow, and avoid "
            "unnecessary leaf wetness."
        )

    if "blight" in name:
        return (
            "Blight is a possible prediction. Check for expanding "
            "dead patches, avoid handling wet plants, and seek "
            "agricultural guidance before selecting treatment."
        )

    if "bacterial" in name:
        return (
            "A bacterial disease is a possible prediction. Avoid "
            "working with wet foliage, clean tools between plants, "
            "and confirm the cause before applying treatment."
        )

    if "mosaic" in name or "virus" in name:
        return (
            "A viral disease is a possible prediction. Check for "
            "mottled or distorted leaves, manage insect vectors "
            "where appropriate, and consult a local agricultural expert."
        )

    return (
        "Inspect the leaf and surrounding plants for similar symptoms. "
        "Use the prediction as a screening aid, not a confirmed diagnosis. "
        "Consult an agricultural expert before treatment."
    )


# --------------------------------------------------
# Page design
# --------------------------------------------------

st.title("🌿 MediX")
st.subheader("AI-Powered Plant Disease Detection")

st.write(
    "Upload a clear plant-leaf image to screen for possible "
    "diseases using the saved MediX deep-learning model."
)

st.info(
    "For best results, use a sharp image with the leaf clearly "
    "visible and good lighting."
)

try:
    with st.spinner("Loading the saved MediX model..."):
        model, class_names = load_medix_model()

    st.success(
        f"Model ready · {len(class_names)} classes"
    )

except Exception as error:
    st.error("MediX could not load its model files.")
    st.code(str(error))
    st.stop()


uploaded_file = st.file_uploader(
    "Choose a plant-leaf image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(
            image,
            caption="Uploaded leaf",
            use_container_width=True,
        )

        if st.button(
            "🔍 Detect Disease",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing the image..."):
                input_array = prepare_image(image, model)
                raw_output = model.predict(
                    input_array,
                    verbose=0,
                )

            predictions = np.asarray(raw_output)

            if predictions.ndim != 2 or predictions.shape[0] != 1:
                st.error(
                    "Unexpected model output. "
                    "This app expects one classification result."
                )
                st.stop()

            scores = predictions[0]

            if len(scores) != len(class_names):
                st.error(
                    "The number of predictions does not match "
                    "the class names."
                )
                st.stop()

            if not np.all(np.isfinite(scores)):
                st.error("The model returned invalid prediction values.")
                st.stop()

            # Only interpret outputs as probabilities if they look
            # like a probability distribution.
            if (
                np.any(scores < 0)
                or np.any(scores > 1)
                or not np.isclose(np.sum(scores), 1.0, atol=0.02)
            ):
                st.warning(
                    "The model outputs do not look like calibrated "
                    "probabilities. The percentages below may not "
                    "represent true confidence."
                )

            ranked_indices = np.argsort(scores)[::-1][:3]
            best_index = int(ranked_indices[0])
            best_label = class_names[best_index]
            best_score = float(scores[best_index])

            st.divider()
            st.header("🌱 Prediction")
            st.markdown(f"**Predicted class:** `{best_label}`")
            st.metric(
                "Model score",
                f"{best_score * 100:.2f}%",
            )

            if best_score >= 0.80:
                st.success(
                    "Strong model score, but the result still needs "
                    "visual and agricultural verification."
                )
            elif best_score >= 0.50:
                st.warning(
                    "Moderate model score. Compare the top predictions "
                    "and verify the symptoms."
                )
            else:
                st.warning(
                    "Low model score. Do not rely on this result alone."
                )

            st.subheader("Top predictions")

            for index in ranked_indices:
                index = int(index)
                label = class_names[index]
                score = float(scores[index])

                st.write(f"**{label}** — {score * 100:.2f}%")
                st.progress(float(np.clip(score, 0.0, 1.0)))

            st.subheader("🌾 General guidance")
            st.write(explain_class(best_label))

            st.caption(
                "MediX is a screening prototype. Predictions may be "
                "incorrect, especially for images from outside the "
                "training dataset. Confirm disease and treatment with "
                "a qualified agricultural expert."
            )

    except Exception as error:
        st.error("Image analysis failed.")
        st.code(str(error))

else:
    st.caption(
        "Your image is processed when you click Detect Disease."
    )

st.divider()
st.caption(
    "MediX · Plant disease screening · Saved model inference"
)
