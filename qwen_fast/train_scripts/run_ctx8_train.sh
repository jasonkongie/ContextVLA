#!/bin/bash
# Fine-tune ContextVLA with ctx=8 on LIBERO using 2x A100.
# Saves checkpoint to /mymnt/checkpoints/ctx8
#
# Usage (from qwen_fast/):
#   bash train_scripts/run_ctx8_train.sh

set -eux

export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false

# Activate conda env (adjust path if miniconda vs anaconda)
source /root/miniforge3/bin/activate contextvla

cd "$(dirname "$0")/.."   # cd to qwen_fast/

mkdir -p /mymnt/checkpoints/ctx8

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=19001 \
    scripts/train_contextvla.py contextvla_libero_ctx8 \
    --exp-name=ctx8_libero \
    --num-train-steps=30000 \
    --batch-size=16 \
    --accum-steps=2 \
    --checkpoint-base-dir=/mymnt/checkpoints/ctx8

echo "ctx=8 training complete. Checkpoint at /mymnt/checkpoints/ctx8"
