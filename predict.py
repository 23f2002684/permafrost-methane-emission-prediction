import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# --- Configuration ---
MODEL_PATH = 'models/methane_classifier_png.pth'
NUM_CLASSES = 3
IMG_SIZE = 224
CLASS_NAMES = {0: "Low Emission Risk", 1: "Moderate Emission Risk", 2: "High Emission Risk"}

# --- Load Model ---
# Create the ResNet18 structure
model = models.resnet18(weights=None) 
# Re-attach our custom final layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Load the weights of YOUR trained model
device = torch.device("cpu")
# --- THIS IS THE LINE TO CHANGE ---
# OLD: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# NEW:
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- Preprocessing and Prediction Function (No changes needed here) ---
def predict_png(image_path):
    """
    Loads a PNG image, preprocesses it, and returns the predicted class and confidence.
    """
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