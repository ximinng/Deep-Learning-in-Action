# -*- coding: utf-8 -*-
"""
   Description :
   Author :        xxm
"""

import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import torch


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


class CoraData(object):
    """
    该数据集共 2708 个样本点, 每个样本点都是一篇科学论文, 所有样本点被分为 8 个类别,
    类别分别是: 1.基于案例; 2.遗传算法; 3.神经网络; 4.概率方法; 5.强化学习; 6.规则学习; 7.理论
    每篇论文都由一个 1433 维的词向量表示, 所以, 每个样本点具有1433个特征。
    词向量的每个元素都对应一个词, 且该元素只有 0 或 1 两个取值;
    取 0 表示该元素对应的词不在论文中, 取 1 表示在论文中。
    所有的词来源于一个具有 1433 个词的字典。
    每篇论文都至少引用了一篇其他论文, 或者被其他论文引用, 也就是样本点之间存在联系, 没有任何一个样本点与其他样本点完全没联系。
    如果将样本点看做图中的点, 则这是一个连通的图, 不存在孤立点。
    """

    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="cora", rebuild=False):
        """Cora数据，包括数据下载，处理，加载等功能
        当数据的缓存文件存在时，将使用缓存文件，否则将下载、进行处理，并缓存到磁盘

        处理之后的数据可以通过属性 .data 获得，它将返回一个数据对象，包括如下几部分：
            * x: 节点的特征，维度为 2708 * 1433，类型为 np.ndarray
            * y: 节点的标签，总共包括7个类别，类型为 np.ndarray
            * adjacency: 邻接矩阵，维度为 2708 * 2708，类型为 scipy.sparse.coo.coo_matrix
            * train_mask: 训练集掩码向量，维度为 2708，当节点属于训练集时，相应位置为True，否则False
            * val_mask: 验证集掩码向量，维度为 2708，当节点属于验证集时，相应位置为True，否则False
            * test_mask: 测试集掩码向量，维度为 2708，当节点属于测试集时，相应位置为True，否则False

        Args:
        -------
            data_root: string, optional
                存放数据的目录，原始数据路径: {data_root}/raw
                缓存数据路径: {data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集，当设为True时，如果存在缓存数据也会重建数据
            adj_dict: boolean, optional
                是否返回dict类型的邻接矩阵
        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

        self.Data = namedtuple('Data', ['x', 'y', 'adjacency',
                                        'train_mask', 'val_mask', 'test_mask'])

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def process_data(self):
        """
        处理数据，得到节点特征和标签，邻接矩阵，训练集、验证集以及测试集
        引用自：https://github.com/rusty1s/pytorch_geometric
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [
            self.read_data(osp.join(self.data_root, "raw", name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        adjacency = self.build_adjacency(graph)

        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return self.Data(x=x, y=y, adjacency=adjacency,
                         train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def maybe_download(self):
        save_path = os.path.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data(
                    "{}/{}".format(self.download_url, name), save_path)

    @staticmethod
    def build_adjacency(adj_dict):
        """根据邻接表创建邻接矩阵"""
        edge_index = []
        num_nodes = len(adj_dict)
        for src, dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去除重复的边
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                                  shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        """使用不同的方式读取原始数据以进一步处理"""
        name = osp.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3)'
                                 ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
        req = urllib.request.Request(url, headers=headers)
        data = urllib.request.urlopen(req)

        filename = os.path.split(url)[-1]

        with open(os.path.join(save_path, filename), 'wb') as f:
            f.write(data.read())

        return True

    @staticmethod
    def normalization(adjacency):
        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        # (A+I), shape: [2708, 2708]
        adjacency += sp.eye(adjacency.shape[0])
        # 计算度矩阵D, shape: [2708, 1]
        degree = np.array(adjacency.sum(1))
        # D^-0.5
        d_hat = sp.diags(np.power(degree, -0.5).flatten())
        # D^-0.5 * (A+I) * D^-0.5
        return d_hat.dot(adjacency).dot(d_hat).tocoo()


if __name__ == '__main__':
    # 加载 Cora 数据集
    dataset = CoraData(data_root='/root/data/cora').data

    # shape: (2708, 1433)
    node_feature = dataset.x / dataset.x.sum(1, keepdims=True)
    # print(node_feature.shape)

    # shape: (2708,)
    # print(dataset.y.shape)

    # shape: (2708,)
    # print(dataset.train_mask.shape)
    # print(dataset.val_mask.shape)
    # print(dataset.test_mask.shape)

    # shape: (2708, 2708)
    normalize_adjacency = CoraData.normalization(dataset.adjacency)  # 规范化邻接矩阵

    num_nodes, input_dim = node_feature.shape

    # shape: (13264,)
    row = normalize_adjacency.row.shape
    col = normalize_adjacency.col.shape

    # edge_index is used to indicate that one node is connected to another node.
    # COO format with shape: (2, 13264) -- [2, num_edges]
    indices = np.asarray([normalize_adjacency.row, normalize_adjacency.col])

    # Node feature matrix with shape [num_nodes, num_node_features]
    # shape: (13264,)
    values = normalize_adjacency.data

    # shape: (2708, 2708)
    tensor_adjacency = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))
