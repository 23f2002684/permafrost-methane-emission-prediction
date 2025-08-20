from flask import Flask, render_template, request, url_for
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Import UNet and physics_filter from utils file
from utils import UNet, physics_filter

# Initialize Flask app
app = Flask(__name__)

#Folder Configurations
UPLOAD_FOLDER = os.path.join('backend', 'uploads')
STATIC_FOLDER = os.path.join('backend', 'static')
MODEL_PATH = os.path.join('backend', 'model', 'unet_permafrost_focal_loss.pth')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

#Load Trained PyTorch U-Net Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

#Image Preprocessing for PyTorch
def preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy() 
    
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device), original_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'satellite_image' not in request.files:
        return "No file part"
    
    file = request.files['satellite_image']
    if file.filename == '':
        return 'No selected file'

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        input_tensor, original_img = preprocess_image(file_path)

        with torch.no_grad():
            pred_mask_tensor = model(input_tensor)
        
        pred_mask = pred_mask_tensor.squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.05).astype(np.uint8) * 255
        mask_img = Image.fromarray(pred_mask)

        mask_filename = f"mask_{file.filename}"
        mask_path = os.path.join(STATIC_FOLDER, mask_filename)
        mask_img.save(mask_path)

        original_img_path = os.path.join(STATIC_FOLDER, file.filename)
        original_img.save(original_img_path)

        #physics-based analysis
        risk_score, risk_level, extra_message = physics_filter(pred_mask)

        return render_template(
            'result.html',
            original_image=url_for('static', filename=file.filename),
            prediction_mask=url_for('static', filename=mask_filename),
            risk_score=round(risk_score, 2),
            risk_level=risk_level,
            extra_message=extra_message
        )
    return "Error during prediction"

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    app.run(debug=True)
