import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch.onnx

#https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

imagenet_class_index = json.load(open('imagenet_class_index.json'))
# load a model 
model = models.densenet121()
model.eval()

print(dir(model))

### SIZE IMAGE ####
## torch.Size([1, 3, 224, 224]) from the transfomed image in the app

expected_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
# input_names = [ "data" ]
input_names = ["input"]
output_names = [ "output" ]

# just like this is possbile ## 
#torch.onnx.export(model, dummy_input, 'resnet110.onnx', verbose=True)

torch.onnx.export(model, expected_input, 'resnet110.onnx', verbose=True, input_names=input_names, output_names=output_names)

## Check the model 

import onnx
print("Running model checks....")
onnx_model = onnx.load("resnet110.onnx")
onnx.checker.check_model(onnx_model)
##print(onnx.helper.printable_graph(model.graph))

# print("---------------------------------------------------------------")
# print("---------------------------------------------------------------")

# dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
# model = torchvision.models.alexnet(pretrained=True)

# # Providing input and output names sets the display names for values
# # within the model's graph. Setting these does not change the semantics
# # of the graph; it is only for readability.
# #
# # The inputs to the network consist of the flat list of inputs (i.e.
# # the values you would pass to the forward() method) followed by the
# # flat list of parameters. You can partially specify names, i.e. provide
# # a list here shorter than the number of inputs to the model, and we will
# # only set that subset of names, starting from the beginning.
# input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
# output_names = [ "output1" ]

# torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


# # Load the ONNX model
# model = onnx.load("alexnet.onnx")

# # Check that the model is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

# examples 
# alexnet
# https://pytorch.org/docs/stable/onnx.html 

# superresolution
# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

import onnxruntime

print("loading back the model via onnx runtime.")
ort_session = onnxruntime.InferenceSession("resnet110.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

batch_size = 1    # setting batch number to 1
# setting the image input size to 3, 244, 244

x = torch.randn(batch_size, 3, 224, 224)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
    