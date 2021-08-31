#!/usr/bin/env bash

for env_name in $1; do
    echo "=> Running environment ${env_name}"
    for random_seed in 125 126 127 128; do
    #for random_seed in $2; do
        python transfer.py --domain_name ${env_name} --task_name $2 \
        --expert_encoder_type identity --expert_decoder_type identity \
        --encoder_type identity --decoder_type identity --expert_dir save/${env_name}/${random_seed} \
        --work_dir save/${env_name}_$3/${random_seed}/agtr_$4 --seed ${random_seed} --gravity -$3
    done
done
