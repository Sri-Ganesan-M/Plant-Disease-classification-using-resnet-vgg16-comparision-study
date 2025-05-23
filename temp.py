import requests
import json
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore') 
  
# Define a simple CNN model for leaf presence classification
class LeafClassifier(nn.Module):
    def __init__(self):
        super(LeafClassifier, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Linear(512, 1)  # Output one unit for binary classification

    def forward(self, x):
        x = self.features(x)
        return x

# Load the trained model
model = LeafClassifier()
model.load_state_dict(torch.load('leaves_classifier_1.pth'))
model.eval()

# Define a function to predict leaf presence
def predict_leaf_presence(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = torch.sigmoid(model(image))
        prediction = True if output.item() < 0.5 else False
    return prediction

# Example usage:



image_path ='rust_fungus-min_1024x1024.png.webp'
prediction = predict_leaf_presence(image_path)
if(prediction):
    # Open the image file
    with open(
       image_path,
        "rb",
    ) as image_file:
        # Send the POST request with the image file
        response = requests.post(
            "http://localhost:4040/predict", files={"image": image_file}
        )

    # Print the response
    pprint(json.loads(response.text))
else:
    print("leaf not detecrted or pass a quality image.....")
