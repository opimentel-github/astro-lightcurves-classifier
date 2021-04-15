#!/bin/bash
gpu=$1 # 1
#mids='1000-1010'
mids=$2
rm nohup.out
eval '
nohup python train_deep_models.py -mc all_rnn_models -gpu $gpu -mids $mids > /dev/null &
nohup python train_deep_models.py -mc all_tcnn_models -gpu $gpu -mids $mids > /dev/null &
nohup python train_deep_models.py -mc parallel_attn_models_te -gpu $gpu -mids $mids > /dev/null &
nohup python train_deep_models.py -mc serial_attn_models_te -gpu $gpu -mids $mids > /dev/null &
nohup python train_deep_models.py -mc parallel_attn_models_te -gpu $gpu -mids $mids -rsc 1 > /dev/null &
nohup python train_deep_models.py -mc serial_attn_models_te -gpu $gpu -mids $mids -rsc 1 > /dev/null &
'