import torch
from torch import nn
from collections import defaultdict
import numpy as np
import torch.functional as F


class attention_aggregator(nn.Module):

    def __init__(self, n_nodes, input_dimension, device):
        super(attention_aggregator, self).__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.linear = torch.nn.Linear(input_dimension, input_dimension, bias=True)
        self.a = torch.nn.Linear(input_dimension, 1, bias=False)

    def forward(self, walks):
        sp = self.linear(walks)  # n * d X d * d -> n * d
        e = self.a(sp)
        e_softmax = F.F.softmax(e)
        walks_embedding = torch.mm(e_softmax.T, walks)
        return walks_embedding


class attention_aggregator2(nn.Module):

    def __init__(self, n_nodes, input_dimension, device):
        super(attention_aggregator2, self).__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.linear = torch.nn.Linear(input_dimension, input_dimension, bias=True)
        self.a = torch.nn.Linear(input_dimension, 1, bias=False)

    def forward(self, walks):
        sp = self.linear(walks)  # n * num_walks  * embedding
        # print("sp:" + str(sp.shape))
        e = self.a(sp)  # n * num_walks * 1
        # print("e:" + str(e.shape))
        e_softmax = F.F.softmax(e, dim=-2)  # n * num_walks * 1

        # print("e_softmax:" + str(e_softmax.shape))
        e_softmax_mask = e_softmax.expand(-1, -1, sp.shape[2])  # n * num_walks * embedding
        # print(e_softmax_mask.grad)
        # print(e.grad)
        # print("e_softmax_mask:" + str(e_softmax_mask.shape))

        walks_embedding = torch.mul(sp, e_softmax_mask)  # n * num_walks * embedding

        return walks_embedding.sum(dim=-2)
