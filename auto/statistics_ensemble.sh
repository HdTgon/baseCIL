#!/bin/bash

datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun" "aircraftb50" "carsb50" "cifarb50" "cubb100" "foodb50" "imagenetrb100" "ucfb50" "sunb150")

for dataset in "${datasets[@]}"; do
    config_file="./exps/ensemble/${dataset}.json"
        echo "Running with dataset=${dataset}, ensemble prototype and mahala"
        CUDA_VISIBLE_DEVICES=0 python main.py --config "$config_file" --para 0.15
done
