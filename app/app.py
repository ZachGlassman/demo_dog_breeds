import base64
import os
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

BATCH_SIZE = 50
IMAGE_DIM = 224
host = os.getenv('serving_host', 'localhost')
MODEL_URL = f"http://{host}:8501/v1/models/test_model"

app = Flask(__name__)

with open("names.txt") as fp:
    names = [i.rstrip("\n") for i in fp.readlines()]


class ImageProcessor:
    def __init__(self, in_memory):
        self.in_memory = in_memory

    @classmethod
    def from_request(cls, data):
        in_memory = BytesIO()
        data.save(in_memory)
        return cls(in_memory)

    def to_image(self):
        self.in_memory.seek(0)
        return np.array(Image.open(self.in_memory))

    def response_image(self):
        self.in_memory.seek(0)
        return base64.b64encode(self.in_memory.read()).decode("utf-8")


def preprocess_image(image):
    return tf.image.resize_with_pad(image, IMAGE_DIM, IMAGE_DIM)


def predict(data):
    to_predict = preprocess_image(data).numpy().astype(int).tolist()

    r = requests.post(
        f"{MODEL_URL}:predict",
        json={"signature_name": "serving_default", "instances": [to_predict]},
    )
    predictions = r.json()["predictions"][0]
    return sorted(zip(names, predictions), key=lambda x: x[1], reverse=True)[:10]


@app.get("/")
def index():
    return render_template("index.html", predictions=None)


@app.post("/")
def with_prediction():
    image = request.files["image"]
    image_processor = ImageProcessor.from_request(image)
    predictions = predict(image_processor.to_image())
    return render_template(
        "index.html", predictions=predictions, image=image_processor.response_image()
    )
