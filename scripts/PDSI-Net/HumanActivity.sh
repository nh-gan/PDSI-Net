#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

GPU_IDS=${1:-0}

model_name="PDSI-Net"
dataset_root_path="storage/datasets/HumanActivity"
dataset_name="HumanActivity"
seq_len=3000
pred_len=300
enc_in=12
dec_in=12
c_out=12

dm=128
bs=8
lr=0.002
dp=0.1

k_nn=96
n_prop=2
c_dim=64
t_dim=128
mlp_r=4.0
w_c=1.0

model_id="MYTRAIN_PATCH_POINTNET_FOURIER_${dataset_name}_sl${seq_len}_pl${pred_len}_dm${dm}_patch4_dp${dp}_lr${lr}_bs${bs}_itr5"

(
  CUDA_VISIBLE_DEVICES=${GPU_IDS} python main.py \
    --is_training 1 \
    --model_id "$model_id" \
    --model_name "$model_name" \
    --dataset_root_path "$dataset_root_path" \
    --dataset_name "$dataset_name" \
    --features M \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --enc_in "$enc_in" \
    --dec_in "$dec_in" \
    --c_out "$c_out" \
    --loss "MSE" \
    --train_epochs 300 \
    --patience 5 \
    --val_interval 1 \
    --itr 5 \
    --batch_size "$bs" \
    --learning_rate "$lr" \
    --d_model "$dm" \
    --dropout "$dp" \
    --astgi_k_neighbors "$k_nn" \
    --astgi_prop_layers "$n_prop" \
    --astgi_channel_dim "$c_dim" \
    --astgi_time_dim "$t_dim" \
    --astgi_mlp_ratio "$mlp_r" \
    --astgi_channel_dist_weight "$w_c"
) &

wait
