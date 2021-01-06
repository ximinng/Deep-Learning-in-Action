<h1 id="dlic" align="center">Deep Learning in Action</h1>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-success" alt="Pyhton 3">
    </a>
     <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/Pytorch-1.x-success" alt="Pytorch">
    </a>
    <a href="http://www.apache.org/licenses/">
        <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
    </a>
    <a href="https://github.com/ximingxing/Deep-Learning-in-Action/pulls">
        <img src="https://img.shields.io/static/v1.svg?label=Contributions&message=Welcome&color=0059b3&style=flat-square" alt="welcome">
    </a>
</p>

<p align="center">
    <a href="#clipboard-getting-started">Getting Started</a> •
    <a href="#table-of-contents">Table of Contents</a> •
    <a href="#about">About</a> •
    <a href="#acknowledgment">Acknowledgment</a> •
    <a href="#speech_balloon-faq">FAQ</a> •
    <a href="#cite">Citing</a> •
</p>

<h6 align="center">Made by ximing Xing • :milky_way:
<a href="https://ximingxing.github.io/">https://ximingxing.github.io/</a>
</h6>

<p align="center">A collection of various deep learning architectures, models, and tips for PyTorch.</p>

<h2 align="center">:clipboard: Getting Started</h2>

### Overview

### Installation

#### Required PyTorch version

- our code requires `torch >= 1.0`

#### Pip

```shell
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

<h2 align="center">:clipboard: Table of Contents</h2>

* Multilayer Perceptrons
    * [MLP](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-basic.ipynb)

* Convolutional Neural Networks (CNN)
    * [AlexNet](https://github.com/ximingxing/Deep-Learning-in-Action/blob/master/convolutional_neural_networks/model/alexnet.py)
    * [VGG](https://github.com/ximingxing/Deep-Learning-in-Action/blob/master/convolutional_neural_networks/model/vgg.py)

* Recurrent Neural Networks (RNN)

* AutoEncoder (AE)

* Generative Adversarial Networks (GAN)

* Graph Neural Networks (GNN)
    * [Graph Convolutional network (GCN)](https://github.com/ximingxing/Deep-Learning-in-Action/blob/master/graph_neural_networks/gcn/graph_convolutional_neural_network.ipynb)

* Deep Reinforcement Learning
    * [Deep Q Learning (DQN)](https://github.com/ximingxing/Deep-Learning-in-Action/blob/master/deep_reinforcement_learning/DQN.ipynb)

* Tips and Tricks

* Visualization and Interpretation

    * Tensorboard
    * Visdom

<h2 align="center">About</h2>

<div align="center">
    <img style="border-radius: 0.3125em; box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"
    height="400"
    src="https://github.com/ximingxing/Images/raw/master/dlic/mental_model_of_the_learning_process.png">
    <br>
    <div style="border-bottom: 1px solid #d9d9d9;display:inline-block;color: #999;padding: 2px;
    font-style: oblique; font-family: 'Times New Roman'">
    Figure: Multilayer Perceptrons model of the learning process</div>
</div>

<h2 align="center">Acknowledgment</h2>

One of the greatest assets of **Deep Learning** is the community and their contributions. A few of my favourite
resources that pair well with the models and componenets here are listed below.

### # Books

- Delip Rao., & Brain McMahan., (2019). *Natural Language Processing with PyTorch*. Sebastopol: O'Reilly Media,Inc.

- Tariq Rashid., (2020). *Make Your First GAN with Pytorch*. Beijing: Posts & Telecom Press.

### # Posts

- [6 GAN Architectures You Really Should Know](https://neptune.ai/blog/6-gan-architectures)

### # Open-source repos

- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

- [Fast Image Augmentation library](https://github.com/albumentations-team/albumentations)

### # Papers

<p align="right"><a href="#dlic"><sup>▴ Back to top</sup></a></p>

<h2 align="center">:speech_balloon: FAQ</h2>
<p align="right"><a href="#dlic"><sup>▴ Back to top</sup></a></p>

<h2 align="center">:speech_balloon: Citing</h2>
<p align="right"><a href="#dlic"><sup>▴ Back to top</sup></a></p>

#### BibTeX

```bibtex
@misc{xx2019dlic,
      author = {Ximing Xing},
      title = {Deep Learning in Action},
      year = {2019},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ximingxing/Deep-Learning-in-Action}}
}
```