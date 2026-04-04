#!/bin/bash

datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun")
few_shot=(8 16 32 64 128 256)

for dataset in "${datasets[@]}"; do
    config_file="./exps/statistics_part_cov/${dataset}.json"
    for fs in "${few_shot[@]}"; do
        echo "Running with dataset=${dataset}, few_shot=${fs}"
        CUDA_VISIBLE_DEVICES=7 python main.py \
            --config "$config_file" \
            --num_sampled 256 \
            --few_shot "$fs"
    done
done