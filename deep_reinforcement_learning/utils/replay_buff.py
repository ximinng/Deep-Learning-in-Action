#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: Deep-Learning-in-Action
@File: replay_buff.py
@Author: 轩名
@Date: 2021/8/13 4:23 下午
"""

from collections import namedtuple, deque
from typing import Tuple
import numpy as np

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "next_state"]
)


class ReplayBuffer(object):
    r"""
    Replay Buffer for storing past experiences allowing the agent to learn from them
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, sample_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            np.array(next_states)
        )

    def __len__(self) -> int:
        return len(self.buffer)
