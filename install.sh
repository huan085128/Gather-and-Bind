#!/bin/bash

# Create conda environment
conda create -n gather_and_bind python=3.10 -y

# Activate the environment
source activate gather_and_bind

# Install PyTorch 1.12.1 with CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other requirements
pip install -r requirements.txt

# Install spaCy model en_core_web_trf
python -m spacy download en_core_web_trf

# 
mkdir models
