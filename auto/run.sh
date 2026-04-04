#!/bin/bash

datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun" )

for dataset in "${datasets[@]}"
do
    config_file="./exps/joint_training/${dataset}.json"

    echo "Running with dataset=${dataset}"
    CUDA_VISIBLE_DEVICES=7 python main.py --config "$config_file"

done