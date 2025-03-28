# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install huggingface_hub for model downloads
RUN pip3 install --no-cache-dir huggingface_hub

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /data

# Create a directory for models
RUN mkdir -p /data/RoomDreamingModel
RUN mkdir -p /data/ControlNetModel/depth
RUN mkdir -p /data/ControlNetModel/seg

# Download models using a properly formatted script
RUN echo "import sys\nfrom huggingface_hub import snapshot_download\ntry:\n    snapshot_download(repo_id='NTUHCILAB/RoomDreamingModel', local_dir='/data/RoomDreamingModel')\n    snapshot_download(repo_id='lllyasviel/control_v11f1p_sd15_depth', local_dir='/data/ControlNetModel/depth')\n    snapshot_download(repo_id='lllyasviel/control_v11p_sd15_seg', local_dir='/data/ControlNetModel/seg')\nexcept Exception as e:\n    print(f'Error downloading models: {str(e)}')\n    sys.exit(1)" > /tmp/download_models.py && python3 /tmp/download_models.py

# Debug: List downloaded files to verify
RUN find /data -type f -exec ls -lh {} \; > /data_contents.txt

# Final image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Copy preloaded models and debug file
COPY --from=builder /data /data
COPY --from=builder /data_contents.txt /data_contents.txt

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libgl1-mesa-glx libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with specific versions
RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install transformers==4.46.0
RUN pip3 install diffusers==0.31.0
RUN pip3 install controlnet_aux==0.0.6
RUN pip3 install mediapipe==0.10.5
RUN pip3 install xformers==0.0.27
RUN pip3 install accelerate==1.0.1
RUN pip3 install runpod

# Copy application code
COPY . /app
WORKDIR /app

# Preload models at container startup
CMD ["python3", "handler.py"]