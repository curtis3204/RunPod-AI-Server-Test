# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

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

RUN pip install \
requests 

# Install PyTorch ecosystem with specific versions - ensure compatibility
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers==4.46.0
RUN pip install diffusers==0.31.0
RUN pip install controlnet_aux==0.0.6
RUN pip install mediapipe==0.10.5
RUN pip install xformers==0.0.27
RUN pip install accelerate==1.5.1

# Copy application code
COPY . /app
WORKDIR /app

# Preload models at container startup
CMD ["python3", "handler.py"]