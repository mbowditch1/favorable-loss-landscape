#!/usr/bin/env bash
cd ..

python3 main.py \
    --script_name get_prop_optimal.sh \
    --get_prop_optimal \
    --data_dir data \
    --data_matrices data_matrices \
    --eps 1e-7 \
    --relu_tol 5e-5 \
    --deriv_tol 5e-5 \
    --convert_tol 5e-5 \
