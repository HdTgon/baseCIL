#!/bin/bash

#datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun")

#datasets=("aircraftb50" "carsb50" "cifarb50" "cubb100" "foodb50" "imagenetrb100" "ucfb50" "sunb150")
datasets=("cifarb50" "cubb100" "foodb50" "imagenetrb100" "ucfb50" "sunb150")

#paras=(0.1 0.15 0.175 0.2)
paras=(0.15)

for dataset in "${datasets[@]}"; do
    config_file="./exps/ensemble/${dataset}.json"
    for para in "${paras[@]}"; do
        echo "Running with dataset=${dataset}, ensemble prototype and mahala, para=${para}"
        CUDA_VISIBLE_DEVICES=6 python main.py --config "$config_file" --para "$para"
    done
done