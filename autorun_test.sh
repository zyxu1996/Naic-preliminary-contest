#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 test.py --models 'HRNet'
echo 'end_HRNet'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 test.py --models 'DANet'
echo 'end_DANet'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 test.py --models 'DeepLabV3_res50'
echo 'end_DeepLabV3_res50'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 test.py --models 'DeepLabV3_res101'
echo 'end_DeepLabV3_res101'