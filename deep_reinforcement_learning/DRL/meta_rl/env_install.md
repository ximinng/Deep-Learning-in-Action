# Deep Reinforcement Learning

## 安装

下面是安装强化学习算法实验环境的过程:

### gym

#### 安装gym

1. 安装gym:

```shell
pip install `gym[all]`
```

2. 测试代码:

```python
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
```

### MuJoCo(Multi-Joint dynamics withContact)

#### 安装MuJoCo

1. [注册](https://link.zhihu.com/?target=https%3A//www.roboti.us/license.html)并获取使用授权, 需要使用edu邮箱, 免费使用一年
2. License下载`getid_osx`, 这个是用于获取自己电脑的id, 下载之后, 获取的命令如下:

```shell
chmod +x getid_osx
./getid_osx
```

3. 提交之后, 会收到一封邮件, 里面含有两个文件: 一个是`LICENSE.txt`、一个是`mjkey.txt`
4. 下载 [mujoco200 macos](https://www.roboti.us/index.html)
5. 在`home`目录下, 创建隐藏文件夹`.mujoco`, 然后将收到的邮件的内容拷贝到`.mujoco`文件夹中

```shell
mkdir ~/.mujoco
cp mujoco200_macos.zip ~/.mujoco

cd ~/.mujoco
unzip mujoco200_macos.zip

# 无论什么系统都需要将目录更改为 mujoco200, 即去掉系统后缀
mv mujoco200_macos/ mujoco200/ 

cp mjkey.txt ~/.mujoco/
cp mjkey.txt ~/.mujoco/mujoco200/bin
```

添加到环境变量:

```shell
vim ~/.bashrc
```

在文件末尾添加如下内容:

```shell
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MUJOCO_KEY_PATH=~/.mujoco${MUJOCO_KEY_PATH}
```

之后:

```shell
source ~/.bashrc
```

6. 测试如下:

```shell
cd ~/.mujoco/mujoco200/bin
./simulate ../model/humanoid.xml
```

#### macos安装mujoco_py

macos记得在`系统设置/安全性与隐私`里给terminal权限.

```shell
pip install -U 'mujoco-py<2.1,>=2.0'
```

#### linux安装mujoco_py

0. 安装gcc

```shell
sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev
```

1. 从源代码安装

```shell
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
```

2. 测试代码:

```python
import mujoco_py
import os

mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
sim.step()
print(sim.data.qpos)
```

### 环境测试

```shell
python dependent_testing_tool.py
```