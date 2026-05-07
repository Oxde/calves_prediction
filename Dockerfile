FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    TORCH_HOME=/app/.torch_cache

WORKDIR /app

# System deps for OpenCV + scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
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
CMD gunicorn app_v3:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 1 --timeout 180 --preload
