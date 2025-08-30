from flask import Flask, render_template, request, url_for
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from utils import UNet, physics_filter

app = Flask(__name__)

# Resolve paths relative to backend/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Files to provide in backend/model:
STATE_DICT_PATH = os.path.join(MODEL_DIR, 'unet_methane_model_state_dict.pth')  # torch.save(model.state_dict(), ...)
TORCHSCRIPT_PATH = os.path.join(MODEL_DIR, 'unet_methane_model.pt')             # torch.jit.script/trace(...).save(...)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_state_dict_if_compatible(model, path, device, min_overlap=0.6):
    """Load a state_dict only if key overlap with the model is reasonable."""
    if not os.path.isfile(path):
        return None
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or len(state) == 0:
        return None

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    overlap = len(model_keys & ckpt_keys) / max(1, len(model_keys))
    if overlap < min_overlap:
        # Try removing 'module.' prefix (DataParallel), then re-check
        renamed = {k.replace('module.', '', 1): v for k, v in state.items()}
        ckpt_keys2 = set(renamed.keys())
        overlap2 = len(model_keys & ckpt_keys2) / max(1, len(model_keys))
        if overlap2 < min_overlap:
            return None
        state = renamed

    # Load with strict=True now that we believe names match
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model

def load_model(device):
    # 1) Try to load a compatible state_dict into our UNet
    model = UNet(in_channels=3, out_channels=1).to(device).eval()
    model = load_state_dict_if_compatible(model, STATE_DICT_PATH, device)
    if model is not None:
        return model

    # 2) Fallback to TorchScript (no class-name dependency)
    if os.path.isfile(TORCHSCRIPT_PATH):
        scripted = torch.jit.load(TORCHSCRIPT_PATH, map_location=device)
        scripted.eval()
        return scripted.to(device)

    # 3) Neither worked -> clear instructions
    raise FileNotFoundError(
        "No compatible model found.\n"
        f"- Provide state_dict: {STATE_DICT_PATH} (export from Colab with torch.save(model.state_dict(), PATH))\n"
        f"- Or provide TorchScript: {TORCHSCRIPT_PATH} (export with torch.jit.script/trace and .save(PATH))\n"
        "Place the file in backend/model and restart."
    )

model = load_model(device)

# Preprocessing (match training; enable Normalize if used in training)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    original_img = img.copy()
    tensor = preprocess(img).unsqueeze(0).to(device)
    return tensor, original_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@torch.no_grad()
def predict():
    if 'satellite_image' not in request.files:
        return "No file part"
    file = request.files['satellite_image']
    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    input_tensor, original_img = preprocess_image(file_path)

    # Inference: logits -> sigmoid -> binary mask
    logits = model(input_tensor)                 # [1,1,H,W]
    probs = torch.sigmoid(logits)                # [0,1]
    pred = (probs > 0.5).to(torch.uint8) * 255   # 0/255 mask

    pred_mask = pred.squeeze().cpu().numpy()
    mask_img = Image.fromarray(pred_mask)

    mask_filename = f"mask_{file.filename}"
    mask_path = os.path.join(STATIC_FOLDER, mask_filename)
    mask_img.save(mask_path)

    original_img_path = os.path.join(STATIC_FOLDER, file.filename)
    original_img.save(original_img_path)

    risk_score, risk_level, extra_message = physics_filter(pred_mask)

    return render_template(
        'result.html',
        original_image=url_for('static', filename=file.filename),
        prediction_mask=url_for('static', filename=mask_filename),
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        extra_message=extra_message
    )

if __name__ == '__main__':
    app.run(debug=True)
