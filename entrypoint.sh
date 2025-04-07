#!/bin/bash

# Set model storage path from environment variable
MODEL_STORAGE=${MODEL_STORAGE:-"/runpod-volume/models"}

# Check and download missing models
python3 check_models.py --storage ${MODEL_STORAGE}

# Start your application
python3 -u handler.py
