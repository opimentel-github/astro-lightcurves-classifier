#!/bin/bash
SECONDS=0
clear

# mc_gpu="--mc p_rnn_models --gpu 0 --precompute_only 1"

### parallel
# mc_gpu="--mc p_attn_model --gpu 0"
# mc_gpu="--mc p_attn_models_te --gpu 0"
# mc_gpu="--mc p_attn_models_noise --gpu 0"
# mc_gpu="--mc p_attn_models_heads --gpu 0"

# mc_gpu="--mc p_rnn_models --gpu 3"

### serial
mc_gpu="--mc s_attn_model --gpu 0"
# mc_gpu="--mc s_attn_models_te --gpu 3"
# mc_gpu="--mc s_attn_models_noise --gpu 3"
# mc_gpu="--mc s_attn_models_heads --gpu 3"

# mc_gpu="--mc s_rnn_models --gpu 0"

b=129
classifier_mids=2

only_perform_exps=0
bypass_autoencoder=0

pt_balanced_metrics=1
ft_balanced_metrics=1

bypass_synth=0
extras="--batch_size $b --only_perform_exps $only_perform_exps --classifier_mids $classifier_mids --bypass_synth $bypass_synth --bypass_autoencoder $bypass_autoencoder --pt_balanced_metrics $pt_balanced_metrics --ft_balanced_metrics $ft_balanced_metrics"

# for mid in 7000; do
for mid in {1000..1004}; do
	for kf in {0..4}; do
		script="python train_deep_models.py --mid $mid --kf $kf $mc_gpu $extras"
		echo "$script"; eval "$script"
	done
done

mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"
