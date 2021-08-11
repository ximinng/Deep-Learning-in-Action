#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@Project: Deep-Learning-in-Action
@File: dependent_testing_tool.py.py
@Author: xingximing.xxm
@Date: 2021/7/6 11:10 上午
"""


def install_gym_testing():
    import gym

    env = gym.make('Humanoid-v2')

    from gym import envs

    print(envs.registry.all())  # print the available environments

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    for i_episode in range(200):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()  # take a random action
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()


def install_mujoco_py_testing():
    import mujoco_py
    import os

    mj_path, _ = mujoco_py.utils.discover_mujoco()
    xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    print(sim.data.qpos)
    sim.step()
    print(sim.data.qpos)


if __name__ == '__main__':
    try:
        install_gym_testing()
    except RuntimeError as error:
        raise error
    print("pass install gym testing")

    try:
        install_mujoco_py_testing()
    except RuntimeError as error:
        raise error
    print("pass install MUJOCO testing")
