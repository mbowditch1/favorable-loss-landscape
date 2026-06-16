#!/usr/bin/env bash
cd ..

python3 main.py \
    --script_name prop_optimum.sh \
    --prop_optimum_approx \
    --sampling_method uniform \
    --data_matrix_index 0 \
    --n "4,6" \
    --m "10,20,40,80,160,320,640" \
    --d "8,10" \
    --N 20 \
    --beta 0.01 \
    --data_matrix_index 0 \
