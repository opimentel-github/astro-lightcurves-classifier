#!/bin/bash
SECONDS=0
clear

#mc_gpu="--mc parallel_attn_models --gpu -1 --only_attn_exp 1"

#mc_gpu="--mc parallel_attn_models --gpu 0"
#mc_gpu="--mc parallel_rnn_models --gpu 1"
#mc_gpu="--mc serial_attn_models --gpu 0"
mc_gpu="--mc serial_rnn_models --gpu 1"

#for mid in {1000..1002} # [a,b]
for mid in {2000..2002} # [a,b]
do
	for kf in {0..4} # [a,b]
	#for kf in {4..4} # [a,b]
	do
		mid_kf="--mid $mid --kf $kf"
		script="python train_deep_models.py $mc_gpu $mid_kf"
		echo "eval $script"
		eval "$script"
	done
done
mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"