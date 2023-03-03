#!/bin/bash

trap "exit" INT
set -e

# orthogonal projection toy experiment (cf. paper): learn orthogonal projection onto the constraint set
OUT_DIR=results
BATCH_SIZE=256
NUM_EPOCHS=500
LR=1e-4

# python main.py --batch_size $BATCH_SIZE --method barycentric --learning_rate $LR --device 0 --num_epochs $NUM_EPOCHS --box_constraints --verbosity 1 --log_to_file --result_dir $OUT_DIR
python main.py --batch_size $BATCH_SIZE --method walker --learning_rate $LR --device 0 --num_epochs $NUM_EPOCHS  --verbosity 1 --log_to_file --result_dir $OUT_DIR

# python plot_results.py
