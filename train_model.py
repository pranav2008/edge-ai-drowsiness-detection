import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# 1. SETUP - ONLY CLASSES 0, 1, 2, 3
data_dir = "dataset"
# Filter to only keep these specific subfolders
included_classes = ['0_closed_eye', '1_open_eye', '2_no_yawn', '3_yawn']

class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in included_classes)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# 2. FAST DATA LOADING
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = FilteredImageFolder(os.path.join(data_dir, 'train'), data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# 3. ACCURATE MODEL (MobileNetV2)
model = models.mobilenet_v2(pretrained=True)
# Adjust final layer for exactly 4 classes
model.classifier[1] = nn.Linear(model.last_channel, 4)
model = model.to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. TRAINING LOOP (Example: 10 Epochs for speed)
print(f"Starting training on classes: {train_dataset.classes}")
model.train()
for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

# 5. AUTO-SAVE & AUTO-EXPORT
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/drowsiness_v3.pt")

# Export to ONNX immediately so you have a backup
dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
torch.onnx.export(model, dummy_input, "models/drowsiness.onnx")
print("Training Complete. Model and ONNX backup saved in 'models/'")
