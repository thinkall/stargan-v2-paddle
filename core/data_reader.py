# !/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: Li JIANG
"""
PaddlePaddle Implementation of StarGAN-v2
"""

import os
from os.path import join
from pathlib import Path
from itertools import chain
from munch import Munch
import sys
import cv2
import math
import random
import functools
import numpy as np
from PIL import Image, ImageOps
import paddle
import paddle.fluid as fluid

from core.dataset import Dataset
from core.image_folder import ImageFolder
from core.batch_sampler import BatchSampler


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


def _transforms(img, img_size, crop='random', scale_ratio=1.1):
    def _random_crop(img, crop_w, crop_h=0):
        if crop_h == 0:
            crop_h = crop_w
        w, h = img.size[0], img.size[1]
        i = np.random.randint(0, w - crop_w)
        j = np.random.randint(0, h - crop_h)
        return img.crop((i, j, i + crop_w, j + crop_h))

    def _center_crop(img, crop_w, crop_h=0):
        if crop_h == 0:
            crop_h = crop_w
        w, h = img.size[0], img.size[1]
        i = int((w - crop_w) / 2.0)
        j = int((h - crop_h) / 2.0)
        return img.crop((i, j, i + crop_w, j + crop_h))

    def _random_horizon_flip(img):
        i = np.random.rand()
        if i > 0.5:
            img = ImageOps.mirror(img)
        return img

    def _normalization(img):
        img = np.array(img)
        img = img / 255
        img = (img - 0.5) / 0.5
        return img

    img = img.resize((int(img_size*scale_ratio), int(img_size*scale_ratio)))
    img = _random_horizon_flip(img)
    img = _random_crop(img, img_size) if crop=='random' else _center_crop(img, img_size)
    img = _normalization(img)
    img = img.transpose(2, 0, 1)
    return img


class DefaultDataset(Dataset):
    def __init__(self, root, transform=None, img_size=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None
        self.img_size = img_size

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img, self.img_size)
        return img

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(Dataset):
    def __init__(self, root, transform=None, img_size=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.img_size = img_size

    def _make_dataset(self, root):
        domains = os.listdir(root)
        for _ in domains:
            if _[0] == '.':
                domains.remove(_)  # remove .DS_Store or other similar files
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img, self.img_size)
            img2 = self.transform(img2, self.img_size)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def get_train_loader(root, which='source', img_size=256,
                     batch_size=8, prob=0.5, num_workers=4):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    transform = _transforms

    if which == 'source':
        dataset = ImageFolder(root, transform, img_size)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform, img_size)
    else:
        raise NotImplementedError

    return BatchSampler(dataset=dataset,
                        which=which,
                        shuffle=True,
                        batch_size=batch_size,
                        drop_last=True,
                        make_balanced_sampler=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = _transforms

    dataset = DefaultDataset(root, transform=transform, img_size=img_size)

    return BatchSampler(dataset=dataset,
                        which=None,
                        shuffle=shuffle,
                        batch_size=batch_size,
                        drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = _transforms

    dataset = ImageFolder(root, transform, img_size)

    return BatchSampler(dataset=dataset,
                        which='source',
                        shuffle=shuffle,
                        batch_size=batch_size,
                        drop_last=False)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = np.random.randn(x.shape[0], self.latent_dim)
            z_trg2 = np.random.randn(x.shape[0], self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v for k, v in inputs.items()})


if __name__ == '__main__':
    place = paddle.fluid.CUDAPlace(0) if paddle.fluid.is_compiled_with_cuda() else paddle.fluid.CPUPlace()
    data_dir = '/Users/jiangli/Work/projects/github-projects/stargan-v2/stargan-v2-paddle/data_dev/celeba_hq/train'
    data_list = listdir(data_dir)
    print(len(data_list), data_list[0])

    dataset = DefaultDataset(data_dir)
    print(len(dataset), dataset[0])

    dataset = ReferenceDataset(data_dir)
    print(len(dataset), dataset[0], dataset[-1])

    bs = get_train_loader(data_dir, which='source')
    print(type(bs), len(bs))
    for bi in bs:
        print(type(bi), len(bi), bi[0].shape, bi[1].shape, bi[1])
        break

    bs = get_train_loader(data_dir, which='reference')
    print(type(bs), len(bs))
    for bi in bs:
        print(type(bi), len(bi), len(bi[0]))
        break

    bs = get_eval_loader(data_dir)
    print(type(bs), len(bs))
    for bi in bs:
        print(type(bi), len(bi))
        break

    bs = get_test_loader(data_dir)
    print(type(bs), len(bs))
    for bi in bs:
        print(type(bi), len(bi), bi[0].shape, bi[1].shape, bi[1])
        break

    src = get_train_loader(data_dir, which='source')
    ref = get_train_loader(data_dir, which='reference')
    val = get_test_loader(data_dir)
    fetcher = InputFetcher(src, ref, 10, 'train')
    fetcher_val = InputFetcher(val, None, 10, 'val')
    inputs_val = next(fetcher_val)
    inputs = next(fetcher)
    print(type(inputs), inputs.keys())
    print(type(inputs.x_src), inputs.x_src.shape)
    x_real, y_org = inputs.x_src, inputs.y_src
    x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
    z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2
    print(type(x_real), type(y_org), type(z_trg))