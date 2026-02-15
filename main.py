import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path

from pathlib import Path
birkin_path = "Data/birkin/"
other_path = "Data/other/"


#Load Images
image_files = []
image_files.extend(birkin_path.glob('*.jpg'))
image_files.extend(other_path.glob('*.jpg'))

#Load Model
print ("Loading pretrained ResNet50 model")
model = models.resnet50(pretrained=True)
#Remove the final classification layer
model.fc = torch.nn.Identify()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Extracting features from images")
features = []
labels = []

for img_path in image_files:
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            feature = model(img_tensor)

        features.append(feature.numpy().flatten())
        