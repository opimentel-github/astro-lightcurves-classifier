#!/bin/bash
p_gpu=0
s_gpu=1

eval '
nohup python train_deep_models.py -mc parallel_rnn_models -gpu $p_gpu -mids 100-110 > /dev/null &
nohup python train_deep_models.py -mc parallel_rnn_models -gpu $p_gpu -mids 200-210 > /dev/null &
nohup python train_deep_models.py -mc parallel_attn_models -gpu $p_gpu -mids 100-110 > /dev/null &
nohup python train_deep_models.py -mc parallel_attn_models -gpu $p_gpu -mids 200-210 > /dev/null &

nohup python train_deep_models.py -mc serial_rnn_models -gpu $s_gpu -mids 100-110 > /dev/null &
nohup python train_deep_models.py -mc serial_rnn_models -gpu $s_gpu -mids 200-210 > /dev/null &
nohup python train_deep_models.py -mc serial_attn_models -gpu $s_gpu -mids 100-110 > /dev/null &
nohup python train_deep_models.py -mc serial_attn_models -gpu $s_gpu -mids 200-210 > /dev/null &
'