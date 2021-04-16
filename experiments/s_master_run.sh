#!/bin/bash
gpu=$1 # 1
#mids='1000-1010'
mids=$2
rm nohup.out
eval '
nohup python train_deep_models.py -mc serial_rnn_models_dt -gpu $gpu -mids $mids > /dev/null &
nohup python train_deep_models.py -mc serial_tcnn_models_dt -gpu $gpu -mids $mids > /dev/null &
nohup python train_deep_models.py -mc serial_attn_models_te -gpu $gpu -mids $mids > /dev/null &
'