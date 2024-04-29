from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-Requested-With", "Content-Type"],
)

image_shape = [212,212]
classes = ["Combwrench", "Hammer", "Screwdriver", "Wrench"]


MODEL = load_model("tool_classifier_model.keras")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(image: User_Input):
    # Decode the image
    image_bytes = base64.b64decode(image.image)
    image_pil = Image.open(io.BytesIO(image_bytes))

    #image_pil.show() #Show the image

    # Ensure RGB
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    # Resize the image
    resized_image = image_pil.resize((212, 212))
    # Put the image in a numpy array
    resized_image_array = np.array(resized_image)
    # Normalize the image pixels
    resized_image_array = resized_image_array / 255.0
    # Add a batch dimension
    image_batch = np.expand_dims(resized_image_array, axis=0)
    # Predict the image with the model
    predictions = MODEL.predict(image_batch)

    # Maximum confidence and the coreesponding class
    max = 0.0
    max_class = ''
    # Check all classes and see if their confidence is higher than the current max
    for i in range(len(classes)):
        print(classes[i], predictions[0][i])
        if predictions[0][i] > max:
            max = predictions[0][i]
            max_class = classes[i]

    print(predictions[0])

    stats = {}
    stats["prediction"] = {
        "class": max_class,
        "confidence": f"{max:.5f}"
    }

    return stats
