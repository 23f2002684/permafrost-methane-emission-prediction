import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

#Configuration
MODEL_PATH = 'models/methane_classifier_final_500plus.pth'
NUM_CLASSES = 3
IMG_SIZE = 224
CLASS_NAMES = {0: "Low Emission Risk", 1: "Moderate Emission Risk", 2: "High Emission Risk"}

#Load Model
model = models.resnet34(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

device = torch.device("cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

#Preprocessing and Prediction Function
def predict_png(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score