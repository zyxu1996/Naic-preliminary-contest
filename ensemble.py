import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np

from models.hrnet import HRNet
from models.danet import DANet
from models.deeplabv3_res50 import DeepLabV3 as DeepLabV3_res50
from models.deeplabv3_res101 import DeepLabV3 as DeepLabV3_res101

from dataset import potsdam
from seg_metric import SegmentationMetric
import cv2
from mutil_scale_test import MultiEvalModule_Fullimg


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


class params():
    def __init__(self):
        self.number_of_classes = 8
        self.TRAIN_BATCH_SIZE_PER_GPU = 64
        self.VAL_BATCH_SIZE_PER_GPU = 64

        "hrnet48"
        self.STAGE2 = {'NUM_MODULES': 1,
                        'NUM_BRANCHES': 2,
                        'NUM_BLOCKS': [4,4],
                        'NUM_CHANNELS': [48,96],
                        'BLOCK':'BASIC',
                        'FUSE_METHOD': 'SUM'}
        self.STAGE3 = {'NUM_MODULES': 4,
                       'NUM_BRANCHES': 3,
                       'NUM_BLOCKS': [4, 4, 4],
                       'NUM_CHANNELS': [48, 96, 192],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
        self.STAGE4 = {'NUM_MODULES': 3,
                       'NUM_BRANCHES': 4,
                       'NUM_BLOCKS': [4, 4, 4, 4],
                       'NUM_CHANNELS': [48, 96, 192, 384],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--use_normalize", type=bool, default=True)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


args = params()

args2 = parse_args()

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
distributed = True
device = torch.device(('cuda:{}').format(args2.local_rank))

danet = DANet(nclass=8)
deeplab = DeepLabV3_res50(nclass=8)
deeplab_101 = DeepLabV3_res101(nclass=8)
deeplab_101_a = DeepLabV3_res101(nclass=8)
PoseHighResolutionNet = HRNet(args)


if distributed:
    torch.cuda.set_device(args2.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
    )

potsdam_val = potsdam(state='val', mean_std=True)
if distributed:
    val_sampler = DistributedSampler(potsdam_val)
else:
    val_sampler = None
dataloader_val = DataLoader(
    potsdam_val,
    batch_size=args.VAL_BATCH_SIZE_PER_GPU,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=val_sampler)

potsdam_test = potsdam(state='test', mean_std=True)
if distributed:
    test_sampler = DistributedSampler(potsdam_test)
else:
    test_sampler = None
dataloader_test = DataLoader(
    potsdam_test,
    batch_size=args.VAL_BATCH_SIZE_PER_GPU,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=test_sampler)

seg_criterion = nn.CrossEntropyLoss(ignore_index=255)

danet = nn.SyncBatchNorm.convert_sync_batchnorm(danet)
danet = danet.to(device)
danet = nn.parallel.DistributedDataParallel(
    danet, device_ids=[args2.local_rank], output_device=args2.local_rank)

deeplab = nn.SyncBatchNorm.convert_sync_batchnorm(deeplab)
deeplab = deeplab.to(device)
deeplab = nn.parallel.DistributedDataParallel(
    deeplab, device_ids=[args2.local_rank], output_device=args2.local_rank)

deeplab_101 = nn.SyncBatchNorm.convert_sync_batchnorm(deeplab_101)
deeplab_101 = deeplab_101.to(device)
deeplab_101 = nn.parallel.DistributedDataParallel(
    deeplab_101, device_ids=[args2.local_rank], output_device=args2.local_rank)

deeplab_101_a = nn.SyncBatchNorm.convert_sync_batchnorm(deeplab_101_a)
deeplab_101_a = deeplab_101_a.to(device)
deeplab_101_a = nn.parallel.DistributedDataParallel(
    deeplab_101_a, device_ids=[args2.local_rank], output_device=args2.local_rank)

PoseHighResolutionNet = nn.SyncBatchNorm.convert_sync_batchnorm(PoseHighResolutionNet)
PoseHighResolutionNet = PoseHighResolutionNet.to(device)
PoseHighResolutionNet = nn.parallel.DistributedDataParallel(
    PoseHighResolutionNet, device_ids=[args2.local_rank], output_device=args2.local_rank)


def test():
    model = MultiEvalModule_Fullimg(PoseHighResolutionNet, nclass=8)
    model1 = MultiEvalModule_Fullimg(danet, nclass=8)
    model2 = MultiEvalModule_Fullimg(deeplab, nclass=8)
    model3 = MultiEvalModule_Fullimg(deeplab_101, nclass=8)
    model4 = MultiEvalModule_Fullimg(deeplab_101_a, nclass=8)
    PoseHighResolutionNet.eval()
    danet.eval()
    deeplab.eval()
    deeplab_101.eval()
    deeplab_101_a.eval()
    with torch.no_grad():
        model_state_file = "weights/NAIC_hrnet48_usually_mean_std_lr0.01125_epoch750_batchsize24_vaihingen3463_xzy_750.pkl"
        model_state_file1 = "weights/NAIC_danet_cutmix_lr0.0075_epoch700_batchsize16_vaihingen3463_best_result_699.pkl"
        model_state_file2 = "weights/NAIC_deeplabv3_attention_lr0.0075_epoch850_batchsize16_vaihingen3463_best_result_849.pkl"
        model_state_file3 = "weights/NAIC_deeplabv3_res101_lr0.01_epoch400_batchsize16_vaihingen3463_best_result_399.pkl"
        model_state_file4 = "weights/NAIC_deeplabv3_res101_labelsmooth_lr0.01_epoch470_batchsize16_vaihingen3463_best_result_469.pkl"

        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            PoseHighResolutionNet.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file1):
            print('loading checkpoint successfully1')
            checkpoint = torch.load(model_state_file1, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            danet.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file2):
            print('loading checkpoint successfully2')
            checkpoint = torch.load(model_state_file2, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            deeplab.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file3):
            print('loading checkpoint successfully3')
            checkpoint = torch.load(model_state_file3, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            deeplab_101.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file4):
            print('loading checkpoint successfully4')
            checkpoint = torch.load(model_state_file4, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            deeplab_101_a.load_state_dict(checkpoint)

        for i, sample in enumerate(dataloader_test):
            if args2.use_normalize:
                images, img_mean_std, name = sample['image'], sample['image_mean_std'], sample['name']
            else:
                images, name = sample['image'], sample['name']

            """if use normalize:img_mean_std"""
            logits = model(img_mean_std)
            logits1 = model1(images)
            logits2 = model2(images)
            logits3 = model3(images)
            logits4 = model4(images)
            logits = logits + logits1 + logits2 + logits3 + logits4
            print("test:{}/{}".format(i, len(dataloader_test)))

            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy().astype(np.uint16)

            logits = (logits + 1) * 100
            for j in range(logits.shape[0]):
                cv2.imwrite("results/" + name[j] + '.png', logits[j])


def test_on_trainset():
    metric = SegmentationMetric(8)
    PoseHighResolutionNet.eval()
    danet.eval()
    deeplab.eval()
    deeplab_101.eval()
    deeplab_101_a.eval()
    with torch.no_grad():
        model_state_file = "weights/NAIC_hrnet48_usually_mean_std_lr0.01125_epoch750_batchsize24_vaihingen3463_xzy_750.pkl"
        model_state_file1 = "weights/NAIC_danet_cutmix_lr0.0075_epoch700_batchsize16_vaihingen3463_best_result_699.pkl"
        model_state_file2 = "weights/NAIC_deeplabv3_attention_lr0.0075_epoch850_batchsize16_vaihingen3463_best_result_849.pkl"
        model_state_file3 = "weights/NAIC_deeplabv3_res101_lr0.01_epoch400_batchsize16_vaihingen3463_best_result_399.pkl"
        model_state_file4 = "weights/NAIC_deeplabv3_res101_labelsmooth_lr0.01_epoch470_batchsize16_vaihingen3463_best_result_469.pkl"

        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            PoseHighResolutionNet.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file1):
            print('loading checkpoint successfully1')
            checkpoint = torch.load(model_state_file1, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            danet.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file2):
            print('loading checkpoint successfully2')
            checkpoint = torch.load(model_state_file2, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            deeplab.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file3):
            print('loading checkpoint successfully3')
            checkpoint = torch.load(model_state_file3, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            deeplab_101.load_state_dict(checkpoint)

        if os.path.isfile(model_state_file4):
            print('loading checkpoint successfully4')
            checkpoint = torch.load(model_state_file4, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            deeplab_101_a.load_state_dict(checkpoint)

        for i, sample in enumerate(dataloader_val):
            images, img_mean_std, labels = sample['image'], sample['image_mean_std'], sample['label']

            logits = PoseHighResolutionNet(img_mean_std)
            logits1 = danet(images)
            logits2 = deeplab(images)
            logits3 = deeplab_101(images)
            logits4 = deeplab_101_a(images)
            logits = logits + logits1 + logits2 + logits3 + logits4

            logits = logits.argmax(dim=1)
            labels = labels.long().squeeze(1)
            print("test:{}/{}".format(i, len(dataloader_val)))

            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)
        fwiou = metric.Frequency_Weighted_Intersection_over_Union()
        print('FWIOU:{}'.format(fwiou))
        fwiou = reduce_tensor(torch.from_numpy(np.array(fwiou)).to(device))
        print('FWIOUx:{}'.format(fwiou))


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29555')

    test()
    # test_on_trainset()





