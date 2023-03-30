#!/bin/bash
# # This should not be changed
#OAR -q production
# # OAR -t exotic
# # Remove `!` for CPUs only
#OAR -p gpu_model!='NO'
# # Adapt as desired
#OAR -p cluster='grele'
#OAR -l host=1/gpu=1/core=2,walltime=01:00:00
# # File where prompts will be outputted
#OAR -O ./OUT/oar_job.%jobid%.output
# # Files where errors will be outputted
#OAR -E ./OUT/oar_job.%jobid%.error

# This is to activate a python environment with conda
source ~/.bashrc

conda activate se

# Set variables
SEGMENT="$1"
CKPT_PATH="../../pretrained_models/A-VAE/A-VAE.pt"
ALGO_TYPE="peem"
DATA_DIR="/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual/TCD-TIMIT/test_data_NTCD/test_data_5.pkl"
SAVE_ROOT="./results/"

# Run command
python SE_evaluation.py \
    --segment "$SEGMENT" \
    --ckpt_path "$CKPT_PATH" \
    --algo_type "$ALGO_TYPE" \
    --data_dir "$DATA_DIR" \
    --save_root "$SAVE_ROOT"