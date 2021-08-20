#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: Deep-Learning-in-Action
@File: gym_env_test.py
@Author: 轩名
@Date: 2021/8/17 5:07 下午
"""


def gym_CartPole_env_test():
    import gym

    env = gym.make('CartPole-v1')
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # take a random action


def gym_CartPole_random_monitor():
    import gym
    env = gym.make("CartPole-v0")
    # 调用gym的monitor模块生成`recording`文件夹
    # 里面包含agent每个step的信息，以及mp4文件记录agent学习过程
    env = gym.wrappers.Monitor(env, "recording")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f"
          % (total_steps, total_reward))
    env.close()
    env.env.close()


def gym_SuperMarioBros_env_test():
    """
    `pip install gym-super-mario-bros==7.3.0`
    """
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace

    # Initialize Super Mario environment
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


if __name__ == "__main__":
    # CartPole
    # gym_CartPole_env_test()

    gym_CartPole_random_monitor()

    # SuperMarioBros
    # gym_SuperMarioBros_env_test()
