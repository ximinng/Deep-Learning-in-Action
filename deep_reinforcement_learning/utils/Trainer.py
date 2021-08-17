#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: Deep-Learning-in-Action
@File: Trainer.py
@Author: 轩名
@Date: 2021/8/17 5:26 下午
"""

import torch
from typing import Any, List
from abc import abstractmethod
from torch import Tensor

from torch.utils.data import DataLoader
from .dataset import RLDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):

    def __init__(self, config, buffer):
        self.config = config

    def train_dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset, self.hparams.batch_size, pin_memory=True
        )
        return dataloader

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def loss_fn(self, *input: Any, **kwargs) -> Tensor:
        pass
