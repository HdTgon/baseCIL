#!/bin/bash

datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun")

for dataset in "${datasets[@]}"; do
    config_file="./exps/statistics_alldata_imbalance/${dataset}.json"
        echo "Running with dataset=${dataset}, imbalance statistics (old classes num 16)"
        CUDA_VISIBLE_DEVICES=7 python main.py --config "$config_file" --num_sampled 256
done
