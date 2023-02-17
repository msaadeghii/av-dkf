#!/bin/bash

# This is to activate a python environment with conda
source ~/.bashrc

nvidia-smi

source activate se

# Run the training script
python train_model.py ./configs/cfg_A_DKF.ini
