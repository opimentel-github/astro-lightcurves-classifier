#!/bin/bash
gpu=0
mids='1000-1005'
for kf in 0 1
do
txt=""
txt+="nohup python train_deep_models.py -mc parallel_attn_models -kf $kf -gpu $gpu -mids $mids > /dev/null &"
#txt+="nohup python train_deep_models.py -mc parallel_rnn_models -kf $kf -gpu 1 -mids $mids > /dev/null &"
#txt+="nohup python train_deep_models.py -mc serial_attn_models -kf $kf -gpu 2 -mids $mids > /dev/null &"
#txt+="nohup python train_deep_models.py -mc serial_rnn_models -kf $kf -gpu 3 -mids $mids > /dev/null &"

echo "$txt"
eval "$txt"
done