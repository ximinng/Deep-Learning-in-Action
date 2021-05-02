# -*- coding: utf-8 -*-
"""
   Description :   PyTorch implementation of GraphSage.
   Author :        xxm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
         input_dim: 输入特征的维度
         output_dim: 输出特征的维度
         use_bias: 是否使用偏置 (default: {False})
         aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def aggregator(self, method):
        return {
            'mean': lambda x: x.mean(dim=1),
            'sum': lambda x: x.sum(dim=1),
            'max': lambda x: x.max(dim=1)
        }.get(method, lambda x: print("Unknown aggr type, expected sum, max, or mean, but got {}".format(method)))

    def forward(self, neighbor_feature):
        aggr_neighbor = self.aggregator(self.aggr_method)(neighbor_feature)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)

        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class SageGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="sum"):
        """SageGCN层定义
        Args:
        input_dim: 输入特征的维度
        hidden_dim: 隐层特征的维度，
            当aggr_hidden_method=sum, 输出维度为hidden_dim
            当aggr_hidden_method=concat, 输出维度为hidden_dim*2
        activation: 激活函数
        aggr_neighbor_method: 邻居特征聚合方法，["mean", "sum", "max"]
        aggr_hidden_method: 节点特征的更新方法，["sum", "concat"]
        """
        super(SageGCN, self).__init__()

        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method

        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight)

    def hidden_aggregator(self, method):
        return {
            'sum': lambda node_hid, neigh_hid: node_hid + neigh_hid,
            'concat': lambda node_hid, neigh_hid: torch.cat([node_hid, neigh_hid], dim=1),
        }.get(method, lambda x: print("Expected sum or concat, but got {}".format(method)))

    def forward(self, source_node_feats, neighbor_node_feats):
        neighbor_hidden = self.aggregator(neighbor_node_feats)
        node_hidden = torch.matmul(source_node_feats, self.weight)

        hidden = self.hidden_aggregator(self.aggr_hidden_method)(neighbor_hidden, node_hidden)

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}' \
            .format(self.input_dim, output_dim, self.aggr_hidden_method)


class GraphSage(nn.Module):
    pass
