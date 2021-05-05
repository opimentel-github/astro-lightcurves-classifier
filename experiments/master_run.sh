#!/bin/bash
p_gpu=1
s_gpu=1
kf=0

eval '
nohup python train_deep_models.py -mc parallel_rnn_models -kf $kf -gpu $p_gpu -mids 100-110 > /dev/null &
nohup python train_deep_models.py -mc parallel_attn_models -kf $kf -gpu $p_gpu -mids 100-110 > /dev/null &

nohup python train_deep_models.py -mc serial_rnn_models -kf $kf -gpu $s_gpu -mids 100-110 > /dev/null &
nohup python train_deep_models.py -mc serial_attn_models -kf $kf -gpu $s_gpu -mids 100-110 > /dev/null &
'