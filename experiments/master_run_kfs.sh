#!/bin/bashkf
mc=$1
gpu=$2
mids=$3
for kf in 0
#for kf in 0 1 2 3 4
do
SECONDS=0

#cmd="nohup python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf > /dev/null"
cmd="python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf"
#python train_deep_models.py -mc $mc -gpu $gpu -mids $mids -kf $kf
eval "$cmd"

mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"
done