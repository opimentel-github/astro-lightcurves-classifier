#!/bin/bash
mc=$1
gpu=$2
mids=$3
for kf in 0 1 2 3 4
do
#cmd="nohup python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf > /dev/null"
cmd="python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf"
#python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf
eval "$cmd"
done