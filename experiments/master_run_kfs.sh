#!/bin/bash
mc=$1
gpu=$2
mids=$3
for kf in 0 1 2 3 4
do
eval "nohup python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf > /dev/null"
done