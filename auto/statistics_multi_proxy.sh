#!/bin/bash

#datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun")
datasets=("objnet")
nb_proxy=(1 2 3 4 5 6 7 8 9 10)

for dataset in "${datasets[@]}"; do
    config_file="./exps/statistics_multi_proxy/${dataset}.json"
    for nb_proxy in "${nb_proxy[@]}"; do
        echo "Running with dataset=${dataset}, nb_proxy=${nb_proxy}"
        CUDA_VISIBLE_DEVICES=5 python main.py \
            --config "$config_file" \
            --nb_proxy "$nb_proxy"
    done
done
