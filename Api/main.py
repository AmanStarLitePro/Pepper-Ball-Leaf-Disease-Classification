from fastapi import FastAPI, UploadFile, File
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("models/my_model.keras")
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

@app.get("/ping")
async def ping():
    return "Server is Ready"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions[0])
    print(predicted_class, confidence)
    return {
        "class" : predicted_class, 
        "confidence" : float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)