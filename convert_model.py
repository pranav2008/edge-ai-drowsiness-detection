import torch
import torch.nn as nn
from torchvision import models
import os

# 1. Re-create the 4-class architecture
model = models.mobilenet_v2()
model.classifier[1] = nn.Linear(model.last_channel, 4)

# 2. Load your weights from the .pt file
model.load_state_dict(torch.load("models/drowsiness_v3.pt"))

# 3. CRITICAL STEP: Switch to Eval Mode
# This disables Dropout and BatchNormalization layers for inference
model.eval() 
model.to("cuda")

# 4. Re-export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
torch.onnx.export(
    model, 
    dummy_input, 
    "models/drowsiness.onnx",
    export_params=True,
    opset_version=17, # Using a stable opset for TensorRT 10.3
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("Fixed ONNX exported successfully in Eval mode!")
