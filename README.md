# Calf Weight Estimator — Web App

Standalone Flask app that wraps the v3 model (`SAM2 + age → GBR`).

```
photo + age  →  YOLO (find calf)  →  SAM2 (mask)  →  48 features  +  age  →  GBR  →  weight (kg)
```

Cross-validated **MAPE 8.4 %, MAE 10.47 kg, R² 0.933** on 203 measurements (Holstein Friesian, 30 calves).

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app_v3.py
# → open http://localhost:8080
```

First run downloads SAM2 (~150 MB) into `~/.cache/huggingface`. Subsequent runs are fast.

Models load in a **background thread** — the page is reachable in ~100 ms. The UI shows a "Warming up the model…" banner that disappears once the `/ready` endpoint returns `{ready: true}`.

## What you'll see

1. Upload a side-view photo (or take one with your phone — `capture="environment"` opens the rear camera).
2. Enter the calf's age in days.
3. Tap **Estimate weight**.
4. Result: green YOLO box + semi-transparent SAM2 mask, big weight number, ± typical-error range, YOLO/SAM2 confidences.
5. The scan is saved to your browser's `localStorage` and shown in **Recent scans**. Tap any thumbnail to replay it.
6. **📷 Scan new cow** clears the form for the next prediction.

Layout adapts to screen size — single column on mobile, two columns (form left, result right) on screens ≥ 960 px.

## Deploy to Railway

Two ways — pick one.

### Option A — Procfile (simplest)

Railway auto-detects Python via `requirements.txt` and runs `Procfile`. **Cold-start**: SAM2 weights download on first boot (~30-60 s extra). Subsequent boots reuse Railway's build cache.

1. `git init` in this folder, push to GitHub.
2. Connect the repo in Railway.
3. Set memory ≥ 2 GB (SAM2 + torch + sklearn ≈ 1.8 GB resident).
4. Open the auto-generated URL.

### Option B — Dockerfile (recommended for prod)

`Dockerfile` is in this folder. Builds the SAM2 weights **into the image at build time**, so cold-starts are fast (~5 s instead of ~60 s).

Railway picks up the Dockerfile automatically. Same memory note (2 GB+).

### Environment variables

None required. Optional:

| Var | Purpose | Default |
|---|---|---|
| `PORT` | Gunicorn bind port | `8080` (Railway sets this automatically) |
| `HF_HOME` | Hugging Face cache dir | `~/.cache/huggingface` |

## Files

| File | What |
|---|---|
| `app_v3.py` | Flask backend. Background-loads YOLO + SAM2 + GBR; exposes `/`, `/predict`, `/ready` |
| `templates/v3.html` | Single self-contained page (CSS + JS inline). Mobile + desktop layouts. localStorage history. |
| `models/v3/` | `yolo_calf_detector.pt` (~6 MB) + `weight_regressor.joblib` + `feature_scaler.joblib` + `feature_config.txt` |
| `models/v2/` | Legacy fallback YOLO if v3 is missing |
| `requirements.txt` | Python deps for both local and Railway |
| `Procfile` | Railway/Heroku start command |
| `Dockerfile` | Optional containerised deploy with pre-cached SAM2 weights |
| `.gitignore` | Keeps `.venv/`, caches, full-resolution test photos out of git |

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | The HTML UI |
| `GET /ready` | `{ready: bool, progress: str, elapsed_s: float, error: str?}`. Polled by the frontend until ready. |
| `POST /predict` | Multipart form: `file` (image), `age_days` (int 1–365). Returns `{detected, predicted_kg, lower_kg, upper_kg, yolo_conf, sam2_conf, annotated_image}` or `{error}`. |

## Notes

- The **legacy v1 app** (`app.py`) still works against `models/v1/` for historical reference. It uses the old DINOv2 + hip_height pipeline that pre-dates the May 2026 client constraint.
- `test_samples/` (13 hand-picked calves with known weight + age) is gitignored by default — keep the deploy slim. Re-include locally if you want to demo on the deployed app.
- The feature column order in `app_v3.py:SAM2_FEATURE_COLS` **must match** what `notebooks/sam2/run_sam2_local.ipynb` writes into `sam2_features.csv`. If you regenerate features differently, retrain pipeline_v3 and update both lists.
