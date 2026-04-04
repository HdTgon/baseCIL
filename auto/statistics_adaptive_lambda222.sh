#!/bin/bash

datasets=("ucf" "imagenetr")

temps=(0.3 0.35 0.4)
gatings=(0 0.05 0.1)

for dataset in "${datasets[@]}"; do
    config_file="./exps/adaptive_lambda/${dataset}.json"
    for temp in "${temps[@]}"; do
        for gating in "${gatings[@]}"; do
          echo "Running with dataset=${dataset}, temp=${temp}, gating=${gating}"
          CUDA_VISIBLE_DEVICES=3 python main.py --config "$config_file" --temp "$temp" --gating "$gating"
        done
    done
done