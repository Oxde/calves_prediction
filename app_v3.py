"""
Calf Weight Estimator — v3 (production-aligned).

Pipeline:  photo + age  →  YOLO (find box)  →  SAM2 (mask)  →  48 features  +  age  →  GBR  →  weight (kg)

Run locally:
    cd /Users/nikmarf/calves_project/calves_git/calves
    source .venv/bin/activate
    pip install -r app/requirements.txt
    python app/app_v3.py
    open http://localhost:8080

Models load in a background thread so the UI is reachable instantly.
"""

import base64
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / 'models' / 'v3'

# ---------- Background-loaded model state ----------

_state = {
    'ready': False,
    'progress': 'starting',
    'error': None,
    'started_at': time.time(),
    'ready_at': None,
}
_state_lock = threading.Lock()

# Bound to module-level once loaded
yolo = None
sam = None
regressor = None
scaler = None
DEVICE = 'cpu'

# Feature column order — MUST match notebooks/sam2/run_sam2_local.ipynb output
SAM2_FEATURE_COLS = [
    'mask_area_px', 'mask_area_ratio', 'perimeter', 'compactness', 'hull_area',
    'hull_perimeter', 'solidity', 'convexity_defect_area', 'bbox_w_px', 'bbox_h_px',
    'extent', 'aspect_ratio', 'major_axis', 'minor_axis', 'eccentricity', 'orientation',
    'hu1', 'hu2', 'hu3', 'hu4',
    'width_max', 'width_mean', 'width_std', 'width_cv', 'width_25pct', 'width_75pct',
    'mean_r', 'std_r', 'mean_g', 'std_g', 'mean_b', 'std_b',
    'mean_brightness', 'mean_saturation',
    'brightness_p10', 'saturation_p10', 'brightness_p25', 'saturation_p25',
    'brightness_p75', 'saturation_p75', 'brightness_p90', 'saturation_p90',
    'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity',
    'lbp_entropy', 'lbp_uniformity',
]


def set_progress(stage):
    with _state_lock:
        _state['progress'] = stage
    print(f'[warmup] {stage}')


def load_models():
    """Run in a background thread. Sets _state['ready'] = True when done."""
    global yolo, sam, regressor, scaler, DEVICE
    try:
        set_progress('importing torch')
        import torch
        if torch.backends.mps.is_available():
            DEVICE = 'mps'
        elif torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'

        set_progress(f'loading YOLO ({DEVICE})')
        from ultralytics import YOLO
        yolo_v3 = MODELS_DIR / 'yolo_calf_detector.pt'
        yolo_v2 = APP_DIR / 'models' / 'v2' / 'yolo_calf_detector.pt'
        yolo_path = yolo_v3 if yolo_v3.exists() else yolo_v2
        yolo = YOLO(str(yolo_path))

        set_progress('loading SAM2 (first run downloads ~80 MB)')
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        # tiny model: ~80 MB on disk, ~250 MB resident — half the RAM of small
        sam = SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-tiny', device=DEVICE)

        set_progress('loading regressor')
        import joblib
        regressor = joblib.load(MODELS_DIR / 'weight_regressor.joblib')
        scaler = joblib.load(MODELS_DIR / 'feature_scaler.joblib')

        set_progress('ready')
        with _state_lock:
            _state['ready'] = True
            _state['ready_at'] = time.time()
        print(f'[warmup] all models loaded in {_state["ready_at"] - _state["started_at"]:.1f}s')
    except Exception as exc:
        with _state_lock:
            _state['error'] = f'{type(exc).__name__}: {exc}'
        print(f'[warmup] ERROR: {_state["error"]}')


# Kick off the background loader as soon as this module is imported
threading.Thread(target=load_models, daemon=True, name='warmup').start()


# ---------- Feature extraction (lazy imports, runs only after warmup) ----------


def shape_features(mask, img_h, img_w, cv2):
    out = {'mask_area_px': float(mask.sum()),
           'mask_area_ratio': float(mask.sum()) / (img_h * img_w)}
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        for k in ['perimeter', 'compactness', 'hull_area', 'hull_perimeter', 'solidity',
                  'convexity_defect_area', 'bbox_w_px', 'bbox_h_px', 'extent', 'aspect_ratio',
                  'major_axis', 'minor_axis', 'eccentricity', 'orientation',
                  'hu1', 'hu2', 'hu3', 'hu4',
                  'width_max', 'width_mean', 'width_std', 'width_cv', 'width_25pct', 'width_75pct']:
            out[k] = 0.0
        return out
    cnt = max(contours, key=cv2.contourArea)
    out['perimeter'] = float(cv2.arcLength(cnt, closed=True))
    area = cv2.contourArea(cnt)
    out['compactness'] = float(4 * np.pi * area / (out['perimeter'] ** 2)) if out['perimeter'] > 0 else 0.0
    hull = cv2.convexHull(cnt)
    out['hull_area'] = float(cv2.contourArea(hull))
    out['hull_perimeter'] = float(cv2.arcLength(hull, closed=True))
    out['solidity'] = float(area / out['hull_area']) if out['hull_area'] > 0 else 0.0
    out['convexity_defect_area'] = float(out['hull_area'] - area)
    x, y, bw, bh = cv2.boundingRect(cnt)
    out['bbox_w_px'] = float(bw); out['bbox_h_px'] = float(bh)
    out['extent'] = float(area / (bw * bh)) if bw * bh > 0 else 0.0
    out['aspect_ratio'] = float(bw / bh) if bh > 0 else 0.0
    if len(cnt) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        out['major_axis'] = float(max(MA, ma)); out['minor_axis'] = float(min(MA, ma))
        out['eccentricity'] = float(np.sqrt(1 - (out['minor_axis'] / out['major_axis']) ** 2)) if out['major_axis'] > 0 else 0.0
        out['orientation'] = float(angle)
    else:
        out['major_axis'] = out['minor_axis'] = out['eccentricity'] = out['orientation'] = 0.0
    hu = cv2.HuMoments(cv2.moments(mask.astype(np.uint8))).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    for i in range(4):
        out[f'hu{i+1}'] = float(hu_log[i])
    rows = mask.sum(axis=1)
    nonzero = rows[rows > 0]
    if nonzero.size:
        out['width_max'] = float(nonzero.max())
        out['width_mean'] = float(nonzero.mean())
        out['width_std'] = float(nonzero.std())
        out['width_cv'] = float(nonzero.std() / nonzero.mean()) if nonzero.mean() > 0 else 0.0
        out['width_25pct'] = float(np.percentile(nonzero, 25))
        out['width_75pct'] = float(np.percentile(nonzero, 75))
    else:
        for k in ('width_max', 'width_mean', 'width_std', 'width_cv', 'width_25pct', 'width_75pct'):
            out[k] = 0.0
    return out


def color_features(img_rgb, mask, cv2):
    masked = img_rgb[mask.astype(bool)]
    if masked.size == 0:
        return {k: 0.0 for k in [
            'mean_r', 'std_r', 'mean_g', 'std_g', 'mean_b', 'std_b',
            'mean_brightness', 'mean_saturation',
            'brightness_p10', 'saturation_p10', 'brightness_p25', 'saturation_p25',
            'brightness_p75', 'saturation_p75', 'brightness_p90', 'saturation_p90',
        ]}
    out = {
        'mean_r': float(masked[:, 0].mean()), 'std_r': float(masked[:, 0].std()),
        'mean_g': float(masked[:, 1].mean()), 'std_g': float(masked[:, 1].std()),
        'mean_b': float(masked[:, 2].mean()), 'std_b': float(masked[:, 2].std()),
    }
    bright = masked.mean(axis=1)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1][mask.astype(bool)]
    out['mean_brightness'] = float(bright.mean())
    out['mean_saturation'] = float(sat.mean())
    for p in (10, 25, 75, 90):
        out[f'brightness_p{p}'] = float(np.percentile(bright, p))
        out[f'saturation_p{p}'] = float(np.percentile(sat, p))
    return out


def texture_features(img_rgb, mask):
    from skimage import feature as skfeature
    from skimage.color import rgb2gray
    gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
    masked_gray = gray.copy()
    masked_gray[~mask.astype(bool)] = 0
    out = {}
    glcm = skfeature.graycomatrix(masked_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ('contrast', 'correlation', 'energy', 'homogeneity'):
        out[f'glcm_{prop}'] = float(skfeature.graycoprops(glcm, prop)[0, 0])
    lbp = skfeature.local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_inside = lbp[mask.astype(bool)]
    if lbp_inside.size:
        hist, _ = np.histogram(lbp_inside, bins=10, range=(0, 10), density=True)
        out['lbp_entropy'] = float(-(hist * np.log2(hist + 1e-12)).sum())
        out['lbp_uniformity'] = float((hist ** 2).sum())
    else:
        out['lbp_entropy'] = 0.0; out['lbp_uniformity'] = 0.0
    return out


def predict_weight(img_pil, age_days):
    import cv2
    import torch
    img_pil = ImageOps.exif_transpose(img_pil).convert('RGB')
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    res = yolo(img_pil, verbose=False, conf=0.25, iou=0.60, imgsz=960)[0]
    if len(res.boxes) == 0:
        res = yolo(img_pil, verbose=False, conf=0.05, iou=0.70, imgsz=1280, augment=True)[0]
    if len(res.boxes) == 0:
        return {'detected': False,
                'message': 'No calf detected in this photo. Try a clearer side-view shot of the whole calf.',
                'annotated_image': pil_to_data_url(img_pil)}

    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    bi = int(np.argmax(confs))
    box = xyxy[bi].astype(np.float32)
    conf = float(confs[bi])

    with torch.inference_mode():
        sam.set_image(img_np)
        masks, sam_scores, _ = sam.predict(box=box[None, :], multimask_output=False)
    mask = masks[0].astype(np.uint8) if masks.ndim == 3 else masks[0, 0].astype(np.uint8)
    sam_score = float(sam_scores[0])

    feats = {}
    feats.update(shape_features(mask, h, w, cv2))
    feats.update(color_features(img_np, mask, cv2))
    feats.update(texture_features(img_np, mask))
    sam2_vec = np.array([feats[c] for c in SAM2_FEATURE_COLS])

    x = np.concatenate([sam2_vec, [float(age_days)]]).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = float(regressor.predict(x_scaled)[0])

    annotated = render_overlay(img_np, box, mask, conf, cv2)

    return {
        'detected': True,
        'predicted_kg': round(pred, 1),
        'yolo_conf': round(conf, 3),
        'sam2_conf': round(sam_score, 3),
        'mape_pct': 8.4,
        'lower_kg': round(pred * 0.916, 1),
        'upper_kg': round(pred * 1.084, 1),
        'annotated_image': annotated,
    }


def render_overlay(img_rgb, box, mask, conf, cv2):
    overlay = img_rgb.copy()
    green = np.zeros_like(overlay); green[:, :, 1] = 255
    alpha = mask.astype(bool)
    overlay[alpha] = (0.55 * overlay[alpha] + 0.45 * green[alpha]).astype(np.uint8)
    x1, y1, x2, y2 = box.astype(int)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 200, 50), max(3, img_rgb.shape[0] // 300))
    label = f'calf  conf={conf:.0%}'
    font_scale = max(0.6, img_rgb.shape[0] / 1200)
    thickness = max(1, img_rgb.shape[0] // 600)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(overlay, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (50, 200, 50), -1)
    cv2.putText(overlay, label, (x1 + 5, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return pil_to_data_url(Image.fromarray(overlay))


def pil_to_data_url(img_pil, quality=85):
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    return f'data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}'


# ---------- Flask ----------

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('v3.html')


@app.route('/ready')
def ready():
    """Polled by the frontend until models finish warming up."""
    with _state_lock:
        elapsed = time.time() - _state['started_at']
        return jsonify({
            'ready': _state['ready'],
            'progress': _state['progress'],
            'elapsed_s': round(elapsed, 1),
            'error': _state['error'],
        })


@app.route('/predict', methods=['POST'])
def predict():
    if not _state['ready']:
        return jsonify({'error': f'Models still warming up: {_state["progress"]}'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    try:
        age_days = int(request.form.get('age_days', '0'))
    except ValueError:
        return jsonify({'error': 'Age must be a number (days)'}), 400
    if age_days < 1 or age_days > 365:
        return jsonify({'error': 'Age in days should be between 1 and 365'}), 400
    try:
        img = Image.open(file.stream)
        return jsonify(predict_weight(img, age_days))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
