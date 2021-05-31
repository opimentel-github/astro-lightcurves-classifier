#!/bin/bash
SECONDS=0
#mc_gpu="--mc parallel_attn_models --gpu 0"
mc_gpu="--mc parallel_rnn_models --gpu 1"
#mc_gpu="--mc serial_attn_models --gpu 2"
#mc_gpu="--mc serial_rnn_models --gpu 3"
only_attn_exp=0
for mid in {2000..2002} # [a,b]
do
	for kf in {3..4} # [a,b]
	do
		config="--mid $mid --kf $kf"
		script="python train_deep_models.py $mc_gpu $config --only_attn_exp $only_attn_exp"
		echo "eval $script"
		eval "$script"
	done
done
mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"