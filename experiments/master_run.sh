#!/bin/bash
SECONDS=0
clear

mc_gpu="--mc parallel_attn_models --gpu 0 --invert_mpg 0"
# mc_gpu="--mc parallel_rnn_models --gpu 0 --invert_mpg 0"

# mc_gpu="--mc serial_attn_models --gpu 1 --invert_mpg 0"
# mc_gpu="--mc serial_rnn_models --gpu 1 --invert_mpg 0"

b=100
bypass=0
extras="--batch_size $b --only_attn_exp 0 --classifier_mids 5 --bypass $bypass"
# extras="--batch_size $b --only_attn_exp 0 --classifier_mids 5"

for mid in {1000..1002}; do
	for kf in {0..4}; do
	# for kf in 1 0 2 3 4; do
		mid_kf="--mid $mid --kf $kf"
		script="python train_deep_models.py $mc_gpu $extras $mid_kf"
		echo "$script"; eval "$script"
	done
done
mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"