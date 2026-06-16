#!/usr/bin/env bash
cd ..

python3 main.py \
    --script_name activation_pattern_dist.sh \
    --activation_pattern_distribution \
    --rw_sampling \
    --n 7 \
    --N 50000 \
    --d 4 \
