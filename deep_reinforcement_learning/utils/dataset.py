#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: Deep-Learning-in-Action
@File: dataset.py
@Author: 轩名
@Date: 2021/8/13 4:25 下午
"""
from typing import Tuple
from torch.utils.data.dataset import IterableDataset

from .replay_buff import ReplayBuffer


class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, next_states, dones = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], next_states[i], dones[i]
