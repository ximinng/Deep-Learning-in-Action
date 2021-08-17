# Deep Reinforcement Learning

This repository contains PyTorch implementations of deep reinforcement learning algorithms and environments.

## Algorithms Implemented

1. Deep Q Learning (DQN)
2. Double Deep Q Learning (Double-DQN)
3. Dueling Deep Q Learning (Dueling-DQN)
4. Prioritized Experience Replay (PER)
5. Deep Deterministic Policy Gradient (DDPG)
6. Asynchronous Advantage Actor-Critic (A3C)
7. Proximal Policy Optimization(PPO)
8. Soft Actor-Critic (SAC)

## Environments Implemented

1. CartPole (Discrete Actions)

2. MountainCar (Continuous Actions)

## Results

## Usage

The repository's high-level structure is:

```
.
├── README.md
├── dqn.py -- DQN
├── test -- 测试文件夹
│      └── gym_env_test.py -- gym环境测试
├── train.py -- 项目的入口
└── utils
    ├── cnn.py -- 模型架构
    ├── dataset.py -- pytroch RL style dataset
    └── replay_buff.py -- 经验回放
```

### install

```shell
pip install gym
```
