# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean

# Install huggingface_hub for model downloads
RUN pip3 install --no-cache-dir huggingface_hub

# Pre-download models during build
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='NTUHCILAB/RoomDreamingModel', local_dir='/data/RoomDreamingModel'); \
    # snapshot_download(repo_id='lllyasviel/Annotators', local_dir='/data/preprocessor'); \
    # snapshot_download(repo_id='h94/IP-Adapter', local_dir='/data/ip-adapter', allow_patterns='models/*'); \
    snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', local_dir='/data/ControlNetModel/depth'); \
    snapshot_download(repo_id='lllyasviel/control_v11p_sd15_seg', local_dir='/data/ControlNetModel/seg')"

# Final image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Copy preloaded models
COPY --from=builder /data /data

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libgl1-mesa-glx && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean

# Install Python dependencies with specific versions
RUN pip3 install --no-cache-dir \
    requests \
    torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 \
    transformers==4.46.0 \
    diffusers==0.31.0 \
    controlnet_aux==0.0.6 \
    mediapipe==0.10.5 \
    xformers==0.0.27 \
    accelerate==1.0.1 \
    runpod  # Added for RunPod serverless compatibility

# Copy application code
COPY . /app
WORKDIR /app

# Preload models at container startup
CMD ["python3", "handler.py"]