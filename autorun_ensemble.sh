#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 ensemble.py
echo 'end_ensemble'
