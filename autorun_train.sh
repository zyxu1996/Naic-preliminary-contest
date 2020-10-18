CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 train.py --end_epoch 700  --models 'HRNet' --lr 0.01 --train_batchsize 16 --val_batchsize 16
echo 'end_HRNet'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 train.py --end_epoch 700  --models 'DANet' --lr 0.01 --train_batchsize 16 --val_batchsize 16
echo 'end_DANet'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 train.py --end_epoch 700  --models 'DeepLabV3_res50' --lr 0.01 --train_batchsize 16 --val_batchsize 16
echo 'end_DeepLabV3_res50'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3.5 -m torch.distributed.launch --nproc_per_node=4 --master_port 29500 train.py --end_epoch 700  --models 'DeepLabV3_res101' --lr 0.01 --train_batchsize 16 --val_batchsize 16
echo 'end_DeepLabV3_res101'