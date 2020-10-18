###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import math
import torch
import torch.nn.functional as F

from torch.nn.parallel.data_parallel import DataParallel


up_kwargs = {'mode': 'nearest', 'align_corners': None}


def module_inference(module, image, flip=True):
    output = module(image)
    if flip:
        fimg = flip_image(image)
        foutput = module(fimg)
        output += flip_image(foutput)
    return output.exp()

def resize_image(img, h, w, **up_kwargs):
    return F.upsample(img, (h, w), **up_kwargs)

def pad_image(img, crop_size):
    b,c,h,w = img.size()
    assert(c==3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    # pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b,c,h+padh,w+padw)
    # for i in range(c):
        # note that pytorch pad params is in reversed orders
    img_pad[:,:,:,:] = F.pad(img[:,:,:,:], (0, padw, 0, padh), mode='reflect')
    assert(img_pad.size(2)>=crop_size and img_pad.size(3)>=crop_size)
    return img_pad

def crop_image(img, h0, h1, w0, w1):
    return img[:,:,h0:h1,w0:w1]

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)


class MultiEvalModule_Fullimg(DataParallel):
    """Multi-size Segmentation Eavluator"""
    def __init__(self, module, nclass, device_ids=None, flip=True,
                 # scales=[1.0]):
                 # scales=[0.5, 1.0, 1.5, 2.0]):
                 # scales=[0.5, 0.75,1.0,1.25,1.5]):
                 scales=[1.0]):
        super(MultiEvalModule_Fullimg, self).__init__(module, device_ids)
        self.nclass = nclass
        self.base_size = 256
        self.crop_size = 256
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule_Fullimg: base_size {}, crop_size {}'. \
            format(self.base_size, self.crop_size))


    def forward(self, image):
        """Mult-size Evaluation"""
        batch, _, h, w = image.size()

        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch,self.nclass,h,w).zero_().cuda()
        for scale in self.scales:
            crop_size = int(math.ceil(self.crop_size * scale))

            cur_img = resize_image(image, crop_size, crop_size, **up_kwargs)
            outputs = module_inference(self.module, cur_img, self.flip)
            score = resize_image(outputs, h, w, **up_kwargs)
            scores += score

        return scores

