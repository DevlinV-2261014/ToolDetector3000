from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import base64
import numpy as np

class User_Input(BaseModel):
    image: str

app = FastAPI()

image_shape = [212,212]

MODEL = load_model("tool_classifier_model.keras")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(image: User_Input):
    image_bytes = base64.b64decode(image.image)
    image_pil = Image.open(io.BytesIO(image_bytes))
    image_pil.show()

    #prediction = MODEL.predict(image_tensor)
    return {"prediction": "¯\_(ツ)_/¯"}
