#!/bin/bash
#mids='1000-1005'
#mids='2000-2005'
mids='1000-1010'

#txt="bash master_run_kfs.sh parallel_attn_models 0 $mids"
#txt="bash master_run_kfs.sh parallel_rnn_models 1 $mids"
#txt="bash master_run_kfs.sh serial_attn_models 2 $mids"
#txt="bash master_run_kfs.sh serial_rnn_models 3 $mids"

#echo "$txt"
eval "$txt"