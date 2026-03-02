import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Allow your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sneaker-style-showcase.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once
brand_model = tf.keras.models.load_model("brand_model.h5")
auth_model = tf.keras.models.load_model("auth_model.h5")

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "Sneaker Backend Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = preprocess(image)

    brand_pred = brand_model.predict(image)
    brand_class = int(np.argmax(brand_pred))

    auth_pred = auth_model.predict(image)
    confidence = float(auth_pred[0][0])
    authenticity = "Real" if confidence > 0.5 else "Fake"

    return {
        "brand": brand_class,
        "authenticity": authenticity,
        "confidence": round(confidence, 4)
    }
