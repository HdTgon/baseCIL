#!/bin/bash

#datasets=("aircraft" "cars" "cifar" "cub" "food" "imagenetr" "ucf" "sun" )
datasets=("objnet")
values=(16 32 64 128 256)

for dataset in "${datasets[@]}"
do
#    config_file="./exps/statistics_alldata_var/${dataset}.json"
    config_file="./exps/statistics_alldata_cov/${dataset}.json"
#    config_file="./exps/statistics_alldata_proto/${dataset}.json"

    for num in "${values[@]}"
    do
        echo "Running with dataset=${dataset}, num_sampled=${num}"
        CUDA_VISIBLE_DEVICES=7 python main.py --config "$config_file" --num_sampled $num
    done
done