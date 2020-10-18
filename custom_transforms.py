import torch
import random
import numpy as np
import cv2


class Rotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.transpose(image)
            label = cv2.transpose(label)

        return {'image': image, 'label': label}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)

        return {'image': image, 'label': label}


class Cutmix(object):
    def __init__(self, alpha=1.):
        self.algha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def __call__(self, img, label):
        # tensor: B C H W
        if random.random() < 0.5:
            indices = torch.randperm(img.size(0))
            # lam = np.clip(np.random.beta(self.algha, self.algha), 0.3, 0.4)  # lam in (0.3,0.4)
            lam = np.random.beta(self.algha, self.algha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
            new_data = img.clone()
            new_target = label.clone()
            new_data[:, :, bbx1:bbx2, bby1:bby2] = img[indices, :, bbx1:bbx2, bby1:bby2]
            new_target[:, :, bbx1:bbx2, bby1:bby2] = label[indices, :, bbx1:bbx2, bby1:bby2]
            return new_data, new_target
        else:
            return img, label


class Cutmix_edge(object):
    def __init__(self, alpha=1.):
        self.algha = alpha

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def __call__(self, img, label, edge):
        # tensor: B C H W
        if random.random() < 0.5:
            indices = torch.randperm(img.size(0))
            # lam = np.clip(np.random.beta(self.algha, self.algha), 0.3, 0.4)  # lam in (0.3,0.4)
            lam = np.random.beta(self.algha, self.algha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(img.size(), lam)
            new_data = img.clone()
            new_target = label.clone()
            new_edge = edge.clone()
            new_data[:, :, bbx1:bbx2, bby1:bby2] = img[indices, :, bbx1:bbx2, bby1:bby2]
            new_target[:, :, bbx1:bbx2, bby1:bby2] = label[indices, :, bbx1:bbx2, bby1:bby2]
            new_edge[:, :, bbx1:bbx2, bby1:bby2] = edge[indices, :, bbx1:bbx2, bby1:bby2]
            return new_data, new_target, new_edge
        else:
            return img, label, edge


class RandomScaleCrop(object):
    def __init__(self, base_size=256, crop_size=256, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.choice([int(self.base_size * 0.5), int(self.base_size * 0.75), int(self.base_size),
                                    int(self.base_size * 1.25), int(self.base_size * 1.5)])
        w, h = img.shape[0:2]
        # print("img.shape:{}".format(img.shape))
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = cv2.copyMakeBorder(img, 0, padh, 0, padw, borderType=cv2.BORDER_DEFAULT)
            mask = cv2.copyMakeBorder(mask, 0, padh, 0, padw, borderType=cv2.BORDER_DEFAULT)
        # random crop crop_size
        w, h = img.shape[0:2]
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img[x1:x1+self.crop_size, y1:y1+self.crop_size, :]
        mask = mask[x1:x1+self.crop_size, y1:y1+self.crop_size]
        return {'image': img, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.int64).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img, 'label': mask}


class imgToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):

        img = sample['image']
        name = sample['name']
        # plt.imshow(mask)
        # plt.show()

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        return {'image': img, 'name': name}
        
        
class ToTensor_mean_std(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #MEAN = torch.tensor(np.array([0.355, 0.383, 0.359])).float()
        #STD = torch.tensor(np.array([0.205, 0.199, 0.208])).float()
        """usually_use"""
        MEAN = torch.tensor(np.array([0.485, 0.456, 0.406])).float()
        STD = torch.tensor(np.array([0.229, 0.224, 0.225])).float()

        img = sample['image']
        mask = sample['label']
        imgx = np.array(img).astype(np.float32).transpose((2, 0, 1))
        imgx = torch.from_numpy(imgx).float()
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.int64).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        img = img / 255
        img = (img - MEAN) / STD
        img = img * 255
        img = img.permute(2, 0, 1)
        mask = torch.from_numpy(mask).float()

        return {'image': imgx, 'image_mean_std': img, 'label': mask}


class imgToTensor_mean_std(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        """count on NAIC dataset"""
        # MEAN = torch.tensor(np.array([0.355, 0.383, 0.359])).float()
        # STD = torch.tensor(np.array([0.205, 0.199, 0.208])).float()
        """usually_use"""
        MEAN = torch.tensor(np.array([0.485, 0.456, 0.406])).float()
        STD = torch.tensor(np.array([0.229, 0.224, 0.225])).float()

        img = sample['image']
        name = sample['name']
        imgx = np.array(img).astype(np.float32).transpose((2, 0, 1))
        imgx = torch.from_numpy(imgx).float()
        img_mean_std = np.array(img).astype(np.float32)
        img_mean_std = torch.from_numpy(img_mean_std).float()
        img_mean_std = img_mean_std / 255
        img_mean_std = (img_mean_std - MEAN) / STD
        img_mean_std = img_mean_std * 255
        img_mean_std = img_mean_std.permute(2, 0, 1)

        return {'image': imgx, 'image_mean_std': img_mean_std, 'name': name}
