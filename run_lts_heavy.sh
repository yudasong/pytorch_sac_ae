#!/usr/bin/env bash

for env_name in $1; do
    echo "=> Running environment ${env_name}"
    for random_seed in 125 126 127 128; do
    	for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8; do
    #for random_seed in $2; do
	        python transfer.py --domain_name ${env_name} --task_name $2 \
	        --expert_encoder_type identity --expert_decoder_type identity \
	        --encoder_type identity --decoder_type identity --expert_dir save/${env_name}/${random_seed} \
	        --work_dir results/${env_name}_$3/${random_seed}/agtr_lts_${ratio} --seed ${random_seed} --gravity -$3 --lts_ratio ${ratio}
	    done
    done
done
