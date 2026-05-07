"""
Calf Weight Estimation — Local Web App

Usage:
    1. Download models from Google Drive (models/ folder) into app/models/
    2. pip install flask torch torchvision ultralytics joblib pillow
    3. python app.py
    4. Open http://localhost:5000
"""

import os
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from flask import Flask, request, render_template, jsonify
from ultralytics import YOLO
import joblib

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / 'models' / 'v1'

# Load models once at startup
print('Loading models...')
yolo = YOLO(str(MODELS_DIR / 'yolo_calf_detector.pt'))

regressor = joblib.load(str(MODELS_DIR / 'weight_regressor.joblib'))
scaler = joblib.load(str(MODELS_DIR / 'feature_scaler.joblib'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dino = dino.to(device)
dino.requires_grad_(False)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f'Models loaded. Device: {device}')

app = Flask(__name__)


def extract_features(img_pil):
    img_tensor = transform(img_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = dino(img_tensor).cpu().numpy()[0]
    return feat


def detect_and_crop(img_pil):
    tmp_path = APP_DIR / '_temp_upload.jpg'
    img_pil.save(str(tmp_path))

    results = yolo(str(tmp_path), verbose=False)
    boxes = results[0].boxes
    tmp_path.unlink(missing_ok=True)

    w, h = img_pil.size

    if len(boxes) == 0:
        side = min(w, h) * 0.8
        left, top = (w - side) / 2, (h - side) / 2
        crop = img_pil.crop((left, top, left + side, top + side))
        return crop.resize((224, 224)), 'fallback (no calf detected)'

    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    conf = float(boxes.conf[best_idx])

    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * 0.20, bh * 0.20
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = img_pil.crop((x1, y1, x2, y2))
    return crop.resize((224, 224)), f'calf detected ({conf:.0%} confidence)'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        img = Image.open(file.stream)
    except Exception:
        return jsonify({'error': 'Could not read image'})

    crop, detection_status = detect_and_crop(img)
    feat = extract_features(crop)

    hip_height = request.form.get('hip_height_cm')
    age = request.form.get('age_days')
    meta_height = float(hip_height) if hip_height else 105.0
    meta_age = float(age) if age else 120.0

    # Use same features for both views (single photo)
    combined = np.concatenate([feat, feat, [meta_height, meta_age]])
    X_input = scaler.transform(combined.reshape(1, -1))
    predicted = float(regressor.predict(X_input)[0])

    return jsonify({
        'weight_kg': round(predicted, 1),
        'detection': detection_status,
    })


if __name__ == '__main__':
    print('\nStarting server at http://localhost:8080')
    app.run(host='0.0.0.0', port=8080, debug=False)
