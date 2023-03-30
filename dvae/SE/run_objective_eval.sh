#!/bin/bash

# This is to activate a python environment with conda
source ~/.bashrc

conda activate se

# Set variables
ENHANCED_DIR="./results/A-VAE/peem"
DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/test_data_5.pkl"
SAVE_DIR="./results/A-VAE/peem"

# Run command
python objective_evaluation.py \
    --enhanced_dir "$ENHANCED_DIR" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR"