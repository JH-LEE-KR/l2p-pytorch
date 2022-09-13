# --------------------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# --------------------------------------------------------
from configparser import Interpolation
import os
import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Lambda

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import utils

class ContinualDataLoader:
    def __init__(self, args):
        self.args = args
        if not os.path.exists(self.args.data_path):
            os.makedirs(self.args.data_path)
        self.transform_train = build_transform(True, self.args)
        self.transform_val = build_transform(False, self.args)
        self._get_dataset(self.args.dataset)

    def _get_dataset(self, name):
        if name == 'CIFAR100':
            root = self.args.data_path
            self.dataset_train = datasets.CIFAR100(root=root, train = True, download = True, transform = self.transform_train)
            self.dataset_val = datasets.CIFAR100(root =root, train = False, transform = self.transform_val)
            self.args.nb_classes = 100
        
        else:
            raise NotImplementedError(f"Not supported dataset: {self.args.dataset}")
        
    def create_dataloader(self):
        dataloader, class_mask = self.split()
        
        return dataloader, class_mask
    
    def target_transform(self, x):
        # Target transform form splited dataset, 0~9 -> 0~9, 10~19 -> 0~9, 20~29 -> 0~9..
        return x - 10*(x//10)

    def split(self):
        dataloader = []
        labels = [i for i in range(self.args.nb_classes)] # [0, 1, 2, ..., 99]
        
        if self.args.shuffle:
            random.shuffle(labels)
        
        class_mask = list() if self.args.task_inc or self.args.train_mask else None
        
        for _ in range(self.args.num_tasks):
            train_split_indices = []
            test_split_indices = []
            
            scope = labels[:self.args.classes_per_task]
            labels = labels[self.args.classes_per_task:]
            
            if class_mask is not None:
                class_mask.append(scope)

            for k in range(len(self.dataset_train.targets)):
                if int(self.dataset_train.targets[k]) in scope:
                    train_split_indices.append(k)
                    
            for h in range(len(self.dataset_val.targets)):
                if int(self.dataset_val.targets[h]) in scope:
                    test_split_indices.append(h)
            
            # self.dataset_train.target_transform = Lambda(self.target_transform)
            # self.dataset_val.target_transform = Lambda(self.target_transform)

            dataset_train, dataset_val =  Subset(self.dataset_train, train_split_indices), Subset(self.dataset_val, test_split_indices)

            if self.args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            else:
                sampler_train = None
                sampler_val = None
            
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
            )

            dataloader.append({'train': data_loader_train, 'val': data_loader_val})
        
        return dataloader, class_mask


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    
    return transforms.Compose(t)
