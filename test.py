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
import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '29555')


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
    parser.add_argument("--models", type=str, default='DeepLabV3_res50', choices=['HRNet','DANet','DeepLabV3_res50','DeepLabV3_res101'])
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


def get_model():
    models = args2.models
    assert models in ['HRNet', 'DANet', 'DeepLabV3_res50', 'DeepLabV3_res101']
    if models == 'HRNet':
        model = HRNet(args)
    if models == 'DANet':
        model = DANet(nclass=8)
    if models == 'DeepLabV3_res50':
        model = DeepLabV3_res50(nclass=8)
    if models == 'DeepLabV3_res101':
        model = DeepLabV3_res101(nclass=8)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args2.local_rank], output_device=args2.local_rank)
    return model


args = params()
args2 = parse_args()

dataset_dir = "../dataset"

cudnn.benchmark = True
cudnn.deterministic = False
cudnn.enabled = True
distributed = True
device = torch.device(('cuda:{}').format(args2.local_rank))


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


def val(model):
    model.eval()
    metric = SegmentationMetric(8)
    with torch.no_grad():

        if args2.models == 'HRNet':
            model_state_file = "weights/NAIC_hrnet48_usually_mean_std_lr0.01125_epoch750_batchsize24_vaihingen3463_xzy_750.pkl"
        if args2.models == 'DANet':
            model_state_file = "weights/NAIC_danet_cutmix_lr0.0075_epoch700_batchsize16_vaihingen3463_best_result_699.pkl"
        if args2.models == 'DeepLabV3_res50':
            model_state_file = "weights/NAIC_deeplabv3_attention_lr0.0075_epoch850_batchsize16_vaihingen3463_best_result_849.pkl"
        if args2.models == 'DeepLabV3_res101':
            model_state_file = "weights/NAIC_deeplabv3_res101_labelsmooth_lr0.01_epoch470_batchsize16_vaihingen3463_best_result_469.pkl"

        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)

        for i, sample in enumerate(dataloader_val):
            if args2.models == 'HRNet':
                images, labels = sample['image_mean_std'], sample['label']
            else:
                images, labels = sample['image'], sample['label']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            print("test:{}/{}".format(i, len(dataloader_test)))
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)
        fwiou = metric.Frequency_Weighted_Intersection_over_Union()
        print('FWIOU:{}'.format(fwiou))
        fwiou = reduce_tensor(torch.from_numpy(np.array(fwiou)).to(device))
        print('FWIOUx:{}'.format(fwiou))


def mutil_scale_val(model):
    model = MultiEvalModule_Fullimg(model, nclass=8)
    model.eval()
    metric = SegmentationMetric(8)
    with torch.no_grad():

        if args2.models == 'HRNet':
            model_state_file = "weights/NAIC_hrnet48_usually_mean_std_lr0.01125_epoch750_batchsize24_vaihingen3463_xzy_750.pkl"
        if args2.models == 'DANet':
            model_state_file = "weights/NAIC_danet_cutmix_lr0.0075_epoch700_batchsize16_vaihingen3463_best_result_699.pkl"
        if args2.models == 'DeepLabV3_res50':
            model_state_file = "weights/NAIC_deeplabv3_attention_lr0.0075_epoch850_batchsize16_vaihingen3463_best_result_849.pkl"
        if args2.models == 'DeepLabV3_res101':
            model_state_file = "weights/NAIC_deeplabv3_res101_labelsmooth_lr0.01_epoch470_batchsize16_vaihingen3463_best_result_469.pkl"

        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)

        for i, sample in enumerate(dataloader_val):
            if args2.models == 'HRNet':
                images, labels = sample['image_mean_std'], sample['label']
            else:
                images, labels = sample['image'], sample['label']
            images = images.cuda()
            labels = labels.long().squeeze(1)
            logits = model(images)
            print("test:{}/{}".format(i, len(dataloader_test)))
            logits = logits.argmax(dim=1)
            # logits = method1(logits)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)
        fwiou = metric.Frequency_Weighted_Intersection_over_Union()
        print('FWIOU:{}'.format(fwiou))
        fwiou = reduce_tensor(torch.from_numpy(np.array(fwiou)).to(device))
        print('FWIOUx:{}'.format(fwiou))


def test(model):
    model = MultiEvalModule_Fullimg(model, nclass=8)
    model.eval()
    with torch.no_grad():

        if args2.models == 'HRNet':
            model_state_file = "weights/NAIC_hrnet48_usually_mean_std_lr0.01125_epoch750_batchsize24_vaihingen3463_xzy_750.pkl"
        if args2.models == 'DANet':
            model_state_file = "weights/NAIC_danet_cutmix_lr0.0075_epoch700_batchsize16_vaihingen3463_best_result_699.pkl"
        if args2.models == 'DeepLabV3_res50':
            model_state_file = "weights/NAIC_deeplabv3_attention_lr0.0075_epoch850_batchsize16_vaihingen3463_best_result_849.pkl"
        if args2.models == 'DeepLabV3_res101':
            model_state_file = "weights/NAIC_deeplabv3_res101_labelsmooth_lr0.01_epoch470_batchsize16_vaihingen3463_best_result_469.pkl"

        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
            checkpoint = {k.replace('model.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)

        for i, sample in enumerate(dataloader_test):
            if args2.models == 'HRNet':
                images, name = sample['image_mean_std'], sample['name']
            else:
                images, name = sample['image'], sample['name']
            images = images.cuda()
            logits = model(images)
            print("test:{}/{}".format(i, len(dataloader_test)))

            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy().astype(np.uint16)

            logits = (logits + 1) * 100
            for j in range(logits.shape[0]):
                cv2.imwrite("results/" + name[j] + '.png', logits[j])


if __name__ == '__main__':
    logging.basicConfig(filename='NAIC_test.log', level=logging.INFO)
    model = get_model()
    val(model)
    # mutil_scale_val(model)
    # test(model)





