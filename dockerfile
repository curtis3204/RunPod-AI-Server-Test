# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git libgl1-mesa-glx libglib2.0-0 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install transformers==4.46.0 diffusers==0.31.0 controlnet_aux==0.0.6 \
    mediapipe==0.10.5 xformers==0.0.27 accelerate==1.0.1 runpod huggingface_hub

# Set up persistent storage location
ENV MODEL_STORAGE=/runpod-volume/models
RUN mkdir -p ${MODEL_STORAGE}

# Copy application code
COPY . /app
WORKDIR /app

# Add model check script
COPY check_models.py /app/check_models.py

# Entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
