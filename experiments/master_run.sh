#!/bin/bash
kf=0
txt=""

p_gpu=0
txt+="nohup python train_deep_models.py -mc parallel_rnn_models -kf $kf -gpu $p_gpu -mids 100-110 > /dev/null &"
txt+="nohup python train_deep_models.py -mc parallel_attn_models -kf $kf -gpu $p_gpu -mids 100-110 > /dev/null &"

s_gpu=1
txt+="nohup python train_deep_models.py -mc serial_rnn_models -kf $kf -gpu $s_gpu -mids 100-110 > /dev/null &"
txt+="nohup python train_deep_models.py -mc serial_attn_models -kf $kf -gpu $s_gpu -mids 100-110 > /dev/null &"

echo "$txt"
eval "$txt"