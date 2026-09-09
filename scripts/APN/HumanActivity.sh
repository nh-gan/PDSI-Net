#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1
GPU_ID=${1:-0}
use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="python"
else
    launch_command="accelerate launch"
fi

model_name="APN"

dataset_root_path=storage/datasets/HumanActivity
model_id=$model_name
dataset_name=$(basename "$0" .sh) # file name

seq_len=3000
for pred_len in 300; do
    $launch_command main.py \
        --gpu_id $GPU_ID \
        --is_training 1 \
        --loss "MSE_Dual" \
        --use_multi_gpu $use_multi_gpu \
        --dataset_root_path $dataset_root_path \
        --model_id $model_id \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 12 \
        --dec_in 12 \
        --c_out 12 \
        --train_epochs 300 \
        --patience 10 \
        --val_interval 1 \
        --itr 5 \
        --batch_size 32 \
        --learning_rate 0.001
done
