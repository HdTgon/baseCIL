#!/bin/bash

datasets=("aircraft" "cars" "cifar" "cub" "food" "sun")

for dataset in "${datasets[@]}"; do
    config_file="./exps/gmm/${dataset}.json"
        echo "Running with dataset=${dataset}, gmm"
        CUDA_VISIBLE_DEVICES=7 python main.py --config "$config_file"
done
