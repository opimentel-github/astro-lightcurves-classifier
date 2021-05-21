#!/bin/bash
SECONDS=0

mids='1000-1005'
#mids='2000-2005'

#txt="bash master_run_kfs.sh parallel_attn_models 0 $mids"
#txt="bash master_run_kfs.sh parallel_rnn_models 0 $mids"
#txt="bash master_run_kfs.sh serial_attn_models 1 $mids"
txt="bash master_run_kfs.sh serial_rnn_models 1 $mids"
#echo "$txt"
eval "$txt"

mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"