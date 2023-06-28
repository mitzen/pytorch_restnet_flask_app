import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch

import numpy as np

# This model provides a REST API and expose the model by loading it
# https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
# model = models.densenet121(pretrained=True)
# model.eval()

import onnxruntime

print("loading back the model via onnx runtime.")
ort_session = onnxruntime.InferenceSession("resnet110.onnx")

# for name, layer in model.features.named_children():
#     print(name, layer)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    print("tensor from image")
    print(tensor.shape)
    print(type(tensor))
    print("tensor image ends")
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor)}
    outputs = ort_session.run(None, ort_inputs)
    ##outputs = model.forward(tensor)
    
    print("outputs from ort_session run")
    print(type(outputs))
    print(dir(outputs))
    print(outputs)

    # tensor_tensor = torch.stack(outputs)
    # print(type(tensor_tensor))
    # print(dir(tensor_tensor))
    # print(tensor_tensor)

    #my_array = np.array(outputs)
    #tensor_tensor = torch.tensor(my_array)

    #tensor_tensor = torch.tensor(outputs)
    #max_value, max_index = torch.max(tensor_tensor, dim=0)

    #print(max_value)
    #print(max_index)
    tensor_tensor = torch.tensor(outputs[0])
    _, y_hat = tensor_tensor.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
