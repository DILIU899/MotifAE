#! /bin/bash
set -Eeuo pipefail

# activate environment
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate MotifAE


nohup python -u ./MotifAE/train_esm_sae.py > "./logs/train_motifae_${HOSTNAME}.log" 2>&1 &