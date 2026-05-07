FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TORCH_HOME=/app/.torch_cache

WORKDIR /app

# Minimal system deps for opencv-python-headless + scikit-image + SAM2 install:
#   git           — required to pip-install SAM2 from GitHub
#   libglib2.0-0  — opencv-python-headless imports it
#   libxcb1       — opencv-python-headless dlopens it on import (~70 KB, tiny)
#   libgomp1      — OpenMP runtime, used by sklearn / scikit-image
# We deliberately do NOT install libgl1 / libsm6 / libxext6 / libxrender1 —
# they pull in libllvm19 (~26 MB) and mesa, and the headless build doesn't need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 libxcb1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps first (cache layer)
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code
COPY . .

# Pre-warm SAM2 weights into the image so first user request is fast
RUN python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; \
               SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-tiny', device='cpu'); \
               print('SAM2 weights cached.')"

EXPOSE 8080
# Use shell form so ${PORT} is expanded by sh, not passed literally.
# No --preload: it would run module-level code (incl. our background warmup thread)
# in the master before fork — the worker would then start with an unstarted thread
# and the model state would never load.
CMD ["sh", "-c", "gunicorn app_v3:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 1 --timeout 180"]
