#!/bin/bash

datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun" "aircraftb50" "carsb50" "cifarb50" "cubb100" "foodb50" "imagenetrb100" "ucfb50" "sunb150")

for dataset in "${datasets[@]}"; do
    config_file="./exps/visual_classifier/${dataset}.json"
        echo "Running with dataset=${dataset}, visual_classifier"
        CUDA_VISIBLE_DEVICES=7 python main.py --config "$config_file" --num_sampled 256
done
