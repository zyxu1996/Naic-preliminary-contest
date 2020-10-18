import os
import argparse
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
from seg_metric import SegmentationMetric
import random

from models.hrnet import HRNet
from models.danet import DANet
from models.deeplabv3_res50 import DeepLabV3 as DeepLabV3_res50
from models.deeplabv3_res101 import DeepLabV3 as DeepLabV3_res101

from dataset import potsdam
import setproctitle
import time
from custom_transforms import Cutmix, Cutmix_edge
from loss import CrossEntropyLoss, NLLMultiLabelSmooth, edge_weak_loss
import logging


class FullModel(nn.Module):

  def __init__(self, model):
    super(FullModel, self).__init__()
    self.model = model
    self.ce_loss = CrossEntropyLoss()
    self.label_smooth = NLLMultiLabelSmooth()
    self.edge_weak_loss = edge_weak_loss()

  def forward(self, inputs, labels, edge=None, epoch=None, train=True):
    output = self.model(inputs)
    if train:
        if epoch < 400:
            loss = self.ce_loss(output, labels)
        elif epoch < 500:
            loss = self.label_smooth(output, labels)
        else:
            loss = self.edge_weak_loss(output, labels, edge)

        return torch.unsqueeze(loss, 0)
    else:
        return output


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


class params():
    def __init__(self):
        self.number_of_classes = 8

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


def get_params():
    pa = params()
    return pa


def get_model(args, args2, device):
    models = args2.models
    print(models)
    assert models in ['HRNet', 'DANet', 'DeepLabV3_res50', 'DeepLabV3_res101']
    if models == 'HRNet':
        model = HRNet(args)
    if models == 'DANet':
        model = DANet(nclass=8)
    if models == 'DeepLabV3_res50':
        model = DeepLabV3_res50(nclass=8)
    if models == 'DeepLabV3_res101':
        model = DeepLabV3_res101(nclass=8)
    model = FullModel(model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args2.local_rank], output_device=args2.local_rank, find_unused_parameters=True)
    return model


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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train_batchsize", type=int, default=16)
    parser.add_argument("--val_batchsize", type=int, default=16)
    parser.add_argument("--models", type=str, default='DeepLabV3_res50',
                        choices=['HRNet', 'DANet', 'DeepLabV3_res50', 'DeepLabV3_res101'])
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args2 = parser.parse_args()
    return args2


def train():
    args = get_params()

    args2 = parse_args()

    torch.manual_seed(args2.seed)
    torch.cuda.manual_seed(args2.seed)
    random.seed(args2.seed)
    np.random.seed(args2.seed)

    distributed = True

    device = torch.device(('cuda:{}').format(args2.local_rank))

    if distributed:
        torch.cuda.set_device(args2.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
    model = get_model(args, args2, device)
    potsdam_train = potsdam(state='train')
    if distributed:
        train_sampler = DistributedSampler(potsdam_train)
    else:
        train_sampler = None
    dataloader_train = DataLoader(
        potsdam_train,
        batch_size=args2.train_batchsize,
        shuffle=True and train_sampler is None,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    potsdam_val = potsdam(state='val')
    if distributed:
        val_sampler = DistributedSampler(potsdam_val)
    else:
        val_sampler = None
    dataloader_val = DataLoader(
        potsdam_val,
        batch_size=args2.val_batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler)

    cutmix = Cutmix()
    cutmix_edge = Cutmix_edge()

    optimizer = torch.optim.SGD([{'params':
                                      filter(lambda p: p.requires_grad,
                                             model.parameters()),
                                      'lr': args2.lr}],
                                    lr=args2.lr,
                                    momentum=0.9,
                                    weight_decay=0.0005,
                                    nesterov=False,
                                    )

    start = time.time()
    miou = [0]
    best_miou = 0.1
    last_epoch = 0
    test_epoch = args2.end_epoch - 3
    ave_loss = AverageMeter()
    world_size = get_world_size()
    reduced_loss = [0]

    model_state_file = "model/{}_lr{}_epoch{}_batchsize{}_naic.pkl.tar" \
          .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize)
    if os.path.isfile(model_state_file):
        print('loaded successfully')
        logging.info("=> loading checkpoint '{}'".format(model_state_file))
        checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
        checkpoint = {k: v for k, v in checkpoint.items() if not 'loss' in k}
        best_miou = checkpoint['best_mIoU']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(
           model_state_file, checkpoint['epoch']))

    for epoch in range(last_epoch, args2.end_epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        setproctitle.setproctitle("xzy:" + str(epoch) + "/" + "{}".format(args2.end_epoch))
        for i, sample in enumerate(dataloader_train):

            if epoch < 500:
                images, labels = sample['image'], sample['label']
                images, labels = images.to(device), labels.to(device)
                images, labels = cutmix(images, labels)
                labels = labels.long().squeeze(1)
                losses = model(images, labels, epoch=epoch)
            else:
                images, labels, edge = sample['image'], sample['label'], sample['edge']
                images, labels, edge = images.to(device), labels.to(device), edge.to(device).unsqueeze(1)
                images, labels, edge = cutmix_edge(images, labels, edge)
                labels = labels.long().squeeze(1)
                edge = edge.long().squeeze(1)
                losses = model(images, labels, edge, epoch)

            loss = losses.mean()
            ave_loss.update(loss.item())

            lr = adjust_learning_rate(optimizer,
                                      args2.lr,
                                      args2.end_epoch * len(dataloader_train),
                                      i + epoch * len(dataloader_train))
            if i % 50 == 0:
                reduced_loss[0] = ave_loss.average()
                print_loss = reduce_tensor(torch.from_numpy(np.array(reduced_loss)).to(device)).cpu()[0] / world_size

                if args2.local_rank == 0:

                    time_cost = time.time() - start
                    print("epoch:[{}/{}], iter:[{}/{}], loss:{}, time:{}, lr:{}, best_miou:{}".format(epoch,args2.end_epoch,i,len(dataloader_train),print_loss,time_cost,lr,best_miou))
                    logging.info(
                        "epoch:[{}/{}], iter:[{}/{}], loss:{}, time:{}, lr:{}, best_miou:{}, miou:{}".format(epoch, args2.end_epoch, i,
                                                                                                len(dataloader_train),
                                                                                                print_loss, time_cost, lr,
                                                                                                best_miou, miou[0]))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            start = time.clock()

        if epoch > test_epoch:
            OA = validate(dataloader_val, device, model)
            miou = reduce_tensor(OA).cpu()

        if args2.local_rank == 0:
            print(miou)

            if epoch > (test_epoch) and epoch % 5 == 0 and epoch != 0:
                torch.save(model.state_dict(),
                           'model/{}_lr{}_epoch{}_batchsize{}_naic_xzy_{}.pkl'
                           .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, epoch))

            if miou[0] >= best_miou:
                best_miou = miou[0]
                torch.save(model.state_dict(),
                           'model/{}_lr{}_epoch{}_batchsize{}_naic_best_result_{}.pkl'
                           .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, epoch))

            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_miou,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'model/{}_lr{}_epoch{}_batchsize{}_naic.pkl.tar'
                .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize))
    torch.save(model.state_dict(),
                   'model/{}_lr{}_epoch{}_batchsize{}_naic_xzy_{}.pkl'
               .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, args2.end_epoch))

    logging.info("***************super param*****************")
    logging.info("dataset:{} lr:{} epoch:{} batchsize:{} best_OA:{}"
                 .format('naic', args2.lr, args2.end_epoch, args2.train_batchsize, best_miou))
    logging.info("***************end*************************")


def adjust_learning_rate(optimizer, base_lr, max_iters,
        cur_iters, power=0.9):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def validate(dataloader_val, device, model):
    model.eval()
    OA = [0]
    metric = SegmentationMetric(8)
    with torch.no_grad():
        for i, sample in enumerate(dataloader_val):
            images, labels = sample['image'], sample['label']
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)
            logits = model(images, labels, train=False)
            logits = logits.argmax(dim=1)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            metric.addBatch(logits, labels)

    fwiou = metric.Frequency_Weighted_Intersection_over_Union()
    OA = OA + fwiou
    print("OA:{}".format(OA))
    OA = torch.from_numpy(OA).to(device)

    return OA


if __name__ == '__main__':
    logging.basicConfig(filename='NAIC_train.log', level=logging.INFO)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29555')

    cudnn.benchmark = True
    cudnn.enabled = True
    train()




