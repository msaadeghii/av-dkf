#!/bin/bash

# This is to activate a python environment with conda
source ~/.bashrc

nvidia-smi

source activate pytorch_venv

# Now you can do the job e.g.
python train_model.py ./example_configuration/dkf/cfg_dkf.ini
