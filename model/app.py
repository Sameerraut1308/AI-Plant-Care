import os
print(os.listdir())

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")
    
class_names = [
'Apple___Apple_scab',
'Apple___Black_rot',
'Apple___Cedar_apple_rust',
'Apple___healthy',
'Blueberry___healthy',
'Cherry___Powdery_mildew',
'Cherry___healthy',
'Corn___Cercospora_leaf_spot Gray_leaf_spot',
'Corn___Common_rust',
'Corn___Northern_Leaf_Blight',
'Corn___healthy',
'Grape___Black_rot',
'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)',
'Peach___Bacterial_spot',
'Peach___healthy',
'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Raspberry___healthy',
'Soybean___healthy',
'Squash___Powdery_mildew',
'Strawberry___Leaf_scorch',
'Strawberry___healthy',
'Tomato___Bacterial_spot',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot',
'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot',
'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

st.title("🌱 Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # show uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    plant, disease = predicted_class.split("___")

    plant = plant.replace("_"," ")
    disease = disease.replace("_"," ")

    st.success(f"🌿 Plant: {plant}")
    st.warning(f"🦠 Disease: {disease}")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")