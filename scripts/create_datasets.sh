#!/usr/bin/env bash
cd ..

python main.py \
    --script_name create_datasets.sh \
    --create_datasets \
    --n "4,6" \
    --d "3,5" \
    --N 5 \
    --data_gen_method teacher \
    --teacher_size 10 \
    --data_dir data \
    --data_matrices data_matrices \
