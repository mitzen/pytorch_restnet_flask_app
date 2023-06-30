from torchvision import models
import torch.onnx

#https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

imagenet_class_index = json.load(open('imagenet_class_index.json'))
# load a model 
model = models.resnet50(pretrained=True)
model.eval()
print(dir(model))

image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True)

torch_out = model(x)
torch.onnx.export(model,                     # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  "resnet50.onnx",              # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=12,             # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'])    # the model's output names


