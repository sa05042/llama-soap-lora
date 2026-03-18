"""
Global configuration for training and inference.
"""

import os
import torch

# Base model
MODEL_NAME = "meta-llama/Llama-3.2-1B"

# Output directories
OUTPUT_DIR = "./output/llama3.2-soap-lora"
ADAPTER_DIR = "./output/llama3.2-soap-lora-adapter"

# Token lengths
MAX_INPUT_LENGTH = 900
MAX_TARGET_LENGTH = 400
MAX_SEQ_LENGTH = MAX_INPUT_LENGTH + MAX_TARGET_LENGTH

# Training configuration
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 8
EPOCHS = 1
LEARNING_RATE = 2e-4
SEED = 42

# Dataset sampling
TRAIN_SAMPLES = 6000

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)