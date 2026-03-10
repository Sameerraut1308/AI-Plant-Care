import io
import os

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry___Powdery_mildew',
    'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

DISEASE_ACTIONS = {
    True: [
        "Isolate affected plants to prevent spread.",
        "Consult an agricultural expert.",
        "Consider appropriate treatment or fungicide.",
        "Monitor other plants for similar symptoms.",
    ],
    False: [
        "Continue regular watering and care.",
        "Ensure adequate sunlight and nutrients.",
        "Monitor for any changes in appearance.",
        "Maintain good air circulation.",
    ],
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "plant_disease_model.h5")

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict(image_bytes: bytes):
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = round(float(np.max(preds)) * 100, 2)
    plant, condition = CLASS_NAMES[idx].split("___")
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ")
    is_disease = "healthy" not in condition.lower()
    status = "DISEASE DETECTED" if is_disease else "HEALTHY PLANT"
    return plant, condition, confidence, is_disease, status, DISEASE_ACTIONS[is_disease]

st.set_page_config(
    page_title="PlantCare AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

def nav_button(label, key):
    if st.button(label, key=key, use_container_width=True):
        st.session_state.page = label
        st.rerun()

col_logo, col_home, col_analyse, col_about, col_spacer = st.columns([3, 1, 1, 1, 3])
with col_logo:
    st.markdown("**PlantCare AI**")
with col_home:
    nav_button("Home", "nav_home")
with col_analyse:
    nav_button("Analyse Plant", "nav_analyse")
with col_about:
    nav_button("About", "nav_about")

st.markdown("---")

page = st.session_state.page


if page == "Home":
    st.title("AI-Powered Plant Disease Detection")
    st.subheader("Protect your crops with cutting-edge artificial intelligence.")
    st.write(
        "Get instant disease identification for healthier plants and better yields. "
        "Just upload a photo of a plant leaf and our AI model will diagnose it in seconds."
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plant Diseases", "38+")
    c2.metric("Accuracy Rate", "95%")
    c3.metric("Plant Species", "10+")
    c4.metric("Availability", "24/7")

    st.markdown("---")
    st.subheader("Why Choose PlantCare AI?")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.info("**Instant Detection**\n\nResults in seconds from our AI model trained on 87,000+ plant images.")
    with f2:
        st.info("**High Accuracy**\n\nMobileNetV2 with transfer learning for precise identification across species.")
    with f3:
        st.info("**Easy to Use**\n\nUpload a photo and get an instant diagnosis — no technical knowledge needed.")

    st.markdown("---")
    st.success("Ready to protect your plants? Click **Analyse Plant** in the navbar above.")


elif page == "Analyse Plant":
    st.title("Analyse Plant")
    st.write("Upload a clear photo of an affected plant leaf for instant AI diagnosis.")

    st.info(
        "Tips: Use good natural lighting · Focus on the leaf · Keep the camera steady · Avoid blur"
    )

    uploaded_file = st.file_uploader(
        "Choose or drag a plant leaf image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        img_col, info_col = st.columns([3, 2])
        with img_col:
            st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)
        with info_col:
            st.subheader("Image Ready")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write("")
            analyse = st.button("Start Detection", type="primary", use_container_width=True)

        if analyse:
            with st.spinner("Analysing image with AI model..."):
                try:
                    plant, condition, confidence, is_disease, status, actions = predict(
                        uploaded_file.getvalue()
                    )

                    st.markdown("---")
                    st.subheader("Analysis Complete")

                    r1, r2 = st.columns(2)
                    with r1:
                        st.image(uploaded_file, caption="Analysed Image", use_container_width=True)
                    with r2:
                        if is_disease:
                            st.error(status)
                        else:
                            st.success(status)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Plant", plant)
                        m2.metric("Condition", condition)
                        m3.metric("Confidence", f"{confidence}%")

                        st.write("")
                        if is_disease:
                            st.warning("**Treatment Recommendations**")
                        else:
                            st.success("**Keep Your Plant Healthy**")
                        for action in actions:
                            st.write(f"- {action}")

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


elif page == "About":
    st.title("About PlantCare AI")
    st.write("Transforming agriculture with artificial intelligence.")

    st.markdown("---")
    st.subheader("Our Mission")
    st.write(
        "We are committed to revolutionising plant health management through cutting-edge artificial "
        "intelligence. Our goal is to empower farmers, gardeners, and agricultural professionals with "
        "instant, accurate disease detection to protect crops and increase yields."
    )
    st.write(
        "By making advanced AI technology accessible to everyone, we help prevent crop losses, "
        "reduce pesticide use, and promote sustainable farming practices."
    )

    st.markdown("---")
    st.subheader("How It Works")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.info("**1. Upload Image**\n\nTake a photo of your plant leaf showing symptoms or upload an existing image.")
    with s2:
        st.info("**2. AI Analysis**\n\nOur MobileNetV2 model processes the image using deep learning algorithms.")
    with s3:
        st.info("**3. Get Results**\n\nReceive an instant diagnosis with plant type, condition, and recommended actions.")

    st.markdown("---")
    st.subheader("Our AI Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Architecture", "MobileNetV2")
    m2.metric("Training Images", "87,000+")
    m3.metric("Plant Species", "10+")
    m4.metric("Diseases Covered", "38+")

    st.markdown("---")
    st.subheader("Benefits")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.success("**Early Detection**\n\nIdentify diseases before they spread.")
    with b2:
        st.success("**Cost Effective**\n\nReduce crop losses and treatment costs.")
    with b3:
        st.success("**Sustainable**\n\nMinimise unnecessary pesticide use.")
    with b4:
        st.success("**Fast & Accurate**\n\nResults in seconds with deep learning precision.")

st.markdown("---")
st.caption("PlantCare AI · Smart Bridge Hyderabad · 2026")
