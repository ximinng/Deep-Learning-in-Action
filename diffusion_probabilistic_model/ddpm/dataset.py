# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os.path as osp
import torchvision


class MNISTDataset(torchvision.datasets.MNIST):

    def __init__(self, data_path, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        if osp.exists(data_path):
            download = False
        else:
            download = True
        super().__init__(data_path, train=True, download=download, transform=transform)
