import io
import json
from PIL import Image
from flask import Flask, jsonify, request
import torch
import numpy as np
import onnxruntime
from torchvision import models
import torchvision.transforms as T
import time 
import onnxruntime
from onnx import numpy_helper
import time

# This model provides a REST API and expose the model by loading it
# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

app = Flask(__name__)

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("loading back the model via onnx runtime.")
session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['CPUExecutionProvider'])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

latency = []

def exec_prediction_task(session, categories, inputs):
    start = time.time()
    input_arr = inputs.cpu().detach().numpy()
    ort_outputs = session.run([], {'input':input_arr})[0]
    latency.append(time.time() - start)
    output = ort_outputs.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:1]
    score = None
    for catid in top5_catid:
        print(categories[catid], output[catid])
        categoryname = categories[catid]
        scores = output[catid]
        return scores, categoryname

def transform_image(image_bytes):
    my_transforms = T.Compose([T.Resize(255),
                                        T.CenterCrop(224),
                                        T.ToTensor(),
                                        T.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
   input_batch = transform_image(image_bytes)
   return exec_prediction_task(session_fp32, categories, input_batch)
   
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': str(class_id), 'class_name': str(class_name)})

if __name__ == '__main__':
    app.run()
