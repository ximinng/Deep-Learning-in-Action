#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: Deep-Learning-in-Action
@File: dqn.py
@Author: 轩名
@Date: 2021/8/11 3:30 下午
"""
import gym
import numpy as np
from collections import OrderedDict
from typing import Tuple, Iterator
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Q_network, Experience, ReplayBuffer, RLDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(object):

    def __init__(self, env: gym.Env, replay_buff: ReplayBuffer):
        self.env = env
        self.replay_buff = replay_buff
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        r"""Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        r"""Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: policy network
            epsilon: value to determine likelihood of taking a random action
            device: cpu or gpu

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()  # np.random.randint(0, n_actions)
        else:
            state = torch.tensor([self.state]).to(device)

            q_value = net(state)
            _, action = torch.max(q_value, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool, dict]:
        r"""
        Carries out a single interaction step between the agent and the environment
        Args:
            net: policy net
            epsilon: epsilon-greedy
            device:

        Returns:
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # get action from policy
        action = self.get_action(net, epsilon, device)

        # use current action to interact with the environment in the current state
        new_state, reward, done, info = self.env.step(action)

        # add new experience to replay buff
        experience = Experience(self.state, action, reward, new_state, done)
        self.replay_buff.append(experience)

        # update state
        self.state = new_state

        if done:
            self.reset()

        return reward, done, info


class Trainer(object):

    def __init__(self,
                 env: str = "CartPole-v0",
                 total_episode: int = 200,
                 batch_size: int = 16,
                 replay_size: int = 1000,
                 hidden_size: int = 128,
                 learning_rate: float = 1e-2,
                 gamma: float = 0.99,
                 sync_rate: int = 10,
                 warm_start_size: int = 1000,
                 eps_last_frame: int = 1000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.01,
                 episode_length: int = 200,
                 warm_start_steps: int = 1000):
        r"""
        Deep Q Learning model
        Args:
            env:
            batch_size: The amount of data sampled from the replay buff
            replay_size: the size of replay buff
            learning_rate:
            gamma: discount factor
            sync_rate:
            warm_start_size:
            eps_last_frame:
            eps_start:
            eps_end:
            episode_length:
            warm_start_steps: populate replay buff
        """
        super(Trainer, self).__init__()

        self.env = gym.make(env)
        obs_size = self.env.observation_space.shape[0]
        print("obs_size: ", obs_size)
        n_actions = self.env.action_space.n

        self.net = Q_network(n_actions)
        self.target_net = Q_network(n_actions)

        self.optimizer = optim.Adam(self.net.parameters(), learning_rate)

        self.total_episode = total_episode
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.episode_length = episode_length
        self.eps_last_frame = eps_last_frame
        self.batch_size = batch_size
        self.sync_rate = sync_rate
        self.gamma = gamma

        self.buffer = ReplayBuffer(replay_size)
        self.agent = DQNAgent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0

        self.populate(warm_start_steps)

        self.global_step = 0

    def train_dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset, self.batch_size, pin_memory=True
        )
        return dataloader

    def populate(self, steps: int = 1000) -> None:
        r"""
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def loss_fn(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        states, actions, rewards, next_states, dones = batch

        state_action_value = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        # bellman backup
        expected_state_action_values = next_state_values * self.gamma + rewards

        loss = nn.MSELoss()
        return loss(state_action_value, expected_state_action_values)

    def training_step(self):
        dataloader = self.train_dataloader()
        for episode in range(self.total_episode):
            for states, actions, rewards, next_states, dones in dataloader:
                batch = (states, actions, rewards, next_states, dones)

                epsilon = max(self.eps_end,
                              self.eps_start - self.global_step + 1 / self.eps_last_frame)

                # step through environment with agent
                reward, done, info = self.agent.play_step(self.net, epsilon, device)
                self.episode_reward += reward

                # calculates training loss
                loss = self.loss_fn(batch)

                # episode end
                if done:
                    self.total_reward = self.episode_reward
                    self.episode_reward = 0

                # copy net to target_net (soft update)
                if self.global_step % self.sync_rate == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                log = {
                    "total_reward": torch.tensor(self.total_reward).to(device),
                    "reward": torch.tensor(reward).to(device),
                    "train_loss": loss
                }
                status = {
                    "steps": torch.tensor(self.global_step).to(device),
                    "total_reward": torch.tensor(self.total_reward).to(device)
                }


if __name__ == '__main__':
    trainer = Trainer()
