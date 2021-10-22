#!/usr/bin/env bash

for env_name in $1; do
    echo "=> Running environment ${env_name}"
    for random_seed in 129 130; do
    #for random_seed in $2; do
        python train.py --domain_name ${env_name} --task_name $2 \
        --encoder_type identity --decoder_type identity --work_dir ../../../scratch/datasets/s2r/${env_name}/${random_seed} \
        --seed ${random_seed} 
    done
done
