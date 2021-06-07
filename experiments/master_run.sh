#!/bin/bash
SECONDS=0
clear

mc_gpu="--mc parallel_attn_models --gpu 0 --invert_mpg 0"
# mc_gpu="--mc parallel_rnn_models --gpu 0 --invert_mpg 0"

# mc_gpu="--mc serial_attn_models --gpu 1 --invert_mpg 0"
# mc_gpu="--mc serial_rnn_models --gpu 1 --invert_mpg 0"

# extras="--s_precomputed_copies 10 --batch_size 128 --only_attn_exp 0 --classifier_mids 10"
extras="--s_precomputed_copies 1 --batch_size 129 --only_attn_exp 0 --classifier_mids 10"

for mid in {1000..1002} # [a,b]
do
	for kf in {0..4} # [a,b]
	do
		mid_kf="--mid $mid --kf $kf"
		script="python train_deep_models.py $mc_gpu $extras $mid_kf"
		echo "eval $script"
		eval "$script"
	done
done
mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"