#!/bin/bash

datasets=("aircraftb50" "carsb50" "cifarb50" "cubb100" "foodb50" "imagenetrb100" "ucfb50" "sunb150")

for dataset in "${datasets[@]}"; do
    config_file="./exps/gmm_ensemble/${dataset}.json"
    echo "Running with dataset=${dataset}"
    CUDA_VISIBLE_DEVICES=6 python main.py --config "$config_file"
done
