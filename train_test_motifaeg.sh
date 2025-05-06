#! /bin/bash
set -Eeuo pipefail

# activate environment
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate MotifAE

# basic configs
## config name
task_name="sample"

## mode, choose from `train_scratch`, `continue_train`, `test_one`
mode="train_scratch"

## checkpoint step, if we want to continue train from one specific checkpoint / test one specific checkpoint
step=0
chk=""  # TODO: Path to your checkpoint

# if we want to train + test from scratch
if [ "$mode" = "train_scratch" ]; then
nohup python -u ./MotifAE_G/train.py -c "./exp_setting/${task_name}.yaml" --mode train_test > "./logs/train_test_${task_name}_${HOSTNAME}.log" 2>&1 &
fi

# if we want to continue train from one specific checkpoint
if [ "$mode" = "continue_train" ]; then
nohup python -u ./MotifAE_G/train.py -c "./exp_setting/${task_name}.yaml" --mode train_test --new_model_checkpoint "${chk}" > "./logs/continue_train_${step}_test_${task_name}_${HOSTNAME}.log" 2>&1 &
fi

# if we want to test one specific checkpoint
## here we update the new checkpoint to enable testing while not changing the original config file
if [ "$mode" = "test_one" ]; then
nohup python -u ./MotifAE_G/train.py -c "./exp_setting/${task_name}.yaml" --mode test --new_model_checkpoint "${chk}" > "./logs/test_${step}_${task_name}_${HOSTNAME}.log" 2>&1 &
fi
