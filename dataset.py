import os
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import custom_transforms as tr
import tifffile as tiff
import torch


class potsdam(data.Dataset):
    def __init__(self, base_dir='../train', state='train', mean_std=False):
        super(potsdam, self).__init__()
        self.base_dir = base_dir
        self.dataset_dir = self.base_dir
        self.state = state
        self.mean_std = mean_std
        if self.state == 'train':
            "RGB"
            self.image_dir = os.path.join(self.dataset_dir, 'image')
            self.label_dir = os.path.join(self.dataset_dir, 'label')
        elif self.state == 'val':
            "RGB"
            self.image_dir = os.path.join(self.dataset_dir, 'image')
            self.label_dir = os.path.join(self.dataset_dir, 'label')
        elif self.state == 'test':
            # self.image_dir = os.path.join('../image_A')
            self.image_dir = os.path.join('../image_B')

        self.filename_list = os.listdir(self.image_dir)

        self.im_ids = []
        self.images = []
        self.labels = []
        for filename in self.filename_list:
            if self.state == 'train' or self.state == 'val':
                image = os.path.join(self.image_dir, filename.strip())
                label = os.path.join(self.label_dir, filename.strip()[:-4] + ".png")
                self.im_ids.append(filename[:-4])
                self.images.append(image)
                self.labels.append(label)
            elif self.state == 'test':
                image = os.path.join(self.image_dir, filename.strip())
                self.im_ids.append(filename[:-4])
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        if self.state == 'train':
            image, label = self.make_img_gt_point_pair(index)
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']
            edge = torch.from_numpy(np.array(Image.fromarray(edge_contour(np.asarray(label))))).long()
            sample = {'image': image, 'label': label, 'edge': edge}
            return sample

        elif self.state == 'val':
            image, label = self.make_img_gt_point_pair(index)
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            return sample

        elif self.state == 'test':
            image = self.make_img_gt_point_pair(index)
            sample = {'image': image, 'name': self.im_ids[index]}
            sample = self.transform(sample)
            return sample

    def make_img_gt_point_pair(self, index):
        if self.state == 'train' or self.state == 'val':
            image = tiff.imread(self.images[index])
            label = Image.open(self.labels[index])
            image = np.array(image)
            label = np.array(label) / 100 - 1
            return image, label
        elif self.state == 'test':
            image = tiff.imread(self.images[index])
            image = np.array(image)
            return image

    def transform(self, sample):

        if self.state == 'train':
            if self.mean_std:
                composed_transforms = transforms.Compose([
                    tr.RandomHorizontalFlip(),
                    tr.RandomVerticalFlip(),
                    tr.RandomScaleCrop(),
                    tr.ToTensor_mean_std()
                ])
            else:
                composed_transforms = transforms.Compose([
                    tr.RandomHorizontalFlip(),
                    tr.RandomVerticalFlip(),
                    tr.RandomScaleCrop(),
                    tr.ToTensor(),
                ])
        elif self.state == 'val':
            if self.mean_std:
                composed_transforms = transforms.Compose([
                    tr.ToTensor_mean_std(),
                ])
            else:
                composed_transforms = transforms.Compose([
                    tr.ToTensor(),
                ])
        elif self.state == 'test':
            if self.mean_std:
                composed_transforms = transforms.Compose([
                    tr.imgToTensor_mean_std(),
                ])
            else:
                composed_transforms = transforms.Compose([
                    tr.imgToTensor(),
                ])
        return composed_transforms(sample)

    def __str__(self):
        if self.state == 'train' or self.state == 'val':
            return 'NAIC(train=True)'
        elif self.state == 'test':
            return 'NAIC(train=False)'


def edge_contour(label, edge_width=3):
    import cv2 as cv

    _, h, w = label.shape
    label = label.squeeze()
    edge = np.zeros(label.shape)

    # right
    edge_right = edge[1:h, :]
    edge_right[(label[1:h, :] != label[:h - 1, :]) & (label[1:h, :] != 255)
               & (label[:h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :w - 1]
    edge_up[(label[:, :w - 1] != label[:, 1:w])
            & (label[:, :w - 1] != 255)
            & (label[:, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:h - 1, :w - 1]
    edge_upright[(label[:h - 1, :w - 1] != label[1:h, 1:w])
                 & (label[:h - 1, :w - 1] != 255)
                 & (label[1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:h - 1, 1:w]
    edge_bottomright[(label[:h - 1, 1:w] != label[1:h, :w - 1])
                     & (label[:h - 1, 1:w] != 255)
                     & (label[1:h, :w - 1] != 255)] = 1

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (edge_width, edge_width))
    edge = cv.dilate(edge, kernel)

    return edge


