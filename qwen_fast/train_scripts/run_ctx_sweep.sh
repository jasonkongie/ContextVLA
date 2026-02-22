#!/bin/bash
# Context-length sweep on Libero: num_frames in {4, 8, 12, 16}
# Runs each experiment sequentially. Each run: 30K steps, batch=16, accum=2.

source ~/anaconda3/bin/activate contextvla

for CTX in 4 8 12 16; do
    echo "=========================================="
    echo "Starting context-length experiment: ctx${CTX}"
    echo "=========================================="

    torchrun --standalone --nnodes=1 --nproc_per_node=8 --master_port=19001 \
        scripts/train_contextvla.py contextvla_libero_ctx${CTX} \
        --exp-name=ctx_sweep_libero_ctx${CTX} \
        --num-train-steps=30000 \
        --batch-size=16 \
        --accum-steps=2

    echo "Finished ctx${CTX}"
done

echo "All context-length sweep experiments complete."
