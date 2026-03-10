import os
import secrets
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import uvicorn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
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

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")

for d in [STATIC_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(title="PlantCare AI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    return {"status": "ok", "message": "PlantCare AI backend is running"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

def predict_image(img_path: str):
    from tensorflow.keras.preprocessing import image as keras_image
    img = keras_image.load_img(img_path, target_size=(224, 224))
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    raw = CLASS_NAMES[idx]
    plant, condition = raw.split("___")
    plant = plant.replace("_", " ")
    condition = condition.replace("_", " ")
    is_disease = "healthy" not in condition.lower()
    return plant, condition, confidence, is_disease

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    tmp_path = os.path.join(IMAGES_DIR, f"tmp_{secrets.token_hex(6)}.jpg")
    static_filename = f"upload_{secrets.token_hex(8)}.jpg"
    static_path = os.path.join(IMAGES_DIR, static_filename)

    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        Image.open(tmp_path).convert("RGB").save(static_path)
        plant, condition, confidence, is_disease = predict_image(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {
        "plant": plant,
        "condition": condition,
        "confidence": round(confidence * 100, 2),
        "is_disease": is_disease,
        "status": "DISEASE DETECTED" if is_disease else "HEALTHY PLANT",
        "actions": DISEASE_ACTIONS[is_disease],
        "image_url": f"http://localhost:5000/static/images/{static_filename}",
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
