import random

import torch
from torch import nn
from collections import defaultdict
import numpy
from modules.attention_aggregator import attention_aggregator2


class random_walk_embedding(nn.Module):

    def __init__(self, n_nodes, input_dimension, output_dimension, repeat_time=4, device="cpu"):
        super(random_walk_embedding, self).__init__()
        self.w_o = torch.nn.Linear(input_dimension, output_dimension, bias=False)
        # nn.Parameter(torch.randn((input_dimension, output_dimension))).to(device)
        self.w_d = torch.nn.Linear(input_dimension, output_dimension, bias=False)
        # nn.Parameter(torch.randn((input_dimension, output_dimension))).to(device)
        self.n_nodes = n_nodes
        self.epsilon = 1e-5
        self.device = device
        self.repeat_time = repeat_time
        self.nodes = range(n_nodes)
        self.linear1 = torch.nn.Linear(n_nodes, output_dimension, bias=True)
        self.aggregator = attention_aggregator2(n_nodes, output_dimension, device)

    def cal_embedding_processed(self, walks, node_embedding):
        # print(self.linear1.weight.data)
        walks_o = torch.LongTensor(walks[0])
        # print(walks_o.shape)
        walks_d = torch.LongTensor(walks[1])
        # print(walks_o)
        # print(walks_d)
        # print(node_embedding.shape)
        walks_o_emb = node_embedding[walks_o]
        # print(walks_o_emb.shape)
        walks_d_emb = node_embedding[walks_d]  # n * num_walks * len_walks * embedding

        walks_o_fc = self.w_o(walks_o_emb)  # torch.matmul(walks_o_emb, self.w_o)
        # print(walks_o_fc.shape)
        walks_d_fc = self.w_d(walks_d_emb)  # torch.matmul(walks_d_emb, self.w_d)

        walks_fc = torch.cat([walks_o_fc, walks_d_fc], dim=-3)
        # print(walks_fc.shape)
        walks_pool = torch.mean(walks_fc, dim=-2)  # n * num_walks  * embedding
        # print(walks_pool.shape)
        walks_agg = self.aggregator(walks_pool)
        return walks_agg

    '''
    def cal_embedding(self, od_mat, node_embedding):
        walk_embedding = []
        import time
        x = time.time()
        for i in range(0, self.n_nodes): 
            walks = []
            for repeat in range(self.repeat_time):
                now_walk = [i]
                now_embedding = [torch.mm(node_embedding[i].reshape(1, -1), self.w_o)]
                for times in range(1 << (repeat + 1)):
                    if times & 1:
                        now_line = od_mat[:][now_walk[-1]]
                        nex_point = numpy.random.choice(a=self.n_nodes, size=1, replace=False,
                                                        p=(now_line.reshape(-1) + self.epsilon) / sum(
                                                            now_line.reshape(-1) + self.epsilon))
                        now_walk.append(nex_point)
                        now_embedding.append(torch.mm(node_embedding[nex_point], self.w_o))

                    else:
                        now_line = od_mat[now_walk[-1]][:]
                        nex_point = numpy.random.choice(a=self.n_nodes, size=1, replace=False,
                                                        p=(now_line.reshape(-1) + self.epsilon) / sum(
                                                            now_line.reshape(-1) + self.epsilon))
                        now_walk.append(nex_point)
                        now_embedding.append(torch.mm(node_embedding[nex_point], self.w_d))
                now_embedding_torch = torch.cat(now_embedding, dim=0)
                walks.append(torch.mean(now_embedding_torch, dim=0))

            for repeat in range(self.repeat_time):
                now_walk = [i]
                now_embedding = [torch.mm(node_embedding[i].reshape(1, -1), self.w_o)]
                for times in range(1 << (repeat + 1)):
                    if times & 1:
                        now_line = od_mat[now_walk[-1]][:]
                        nex_point = numpy.random.choice(a=self.n_nodes, size=1, replace=False,
                                                        p=(now_line.reshape(-1) + self.epsilon) / sum(
                                                            now_line.reshape(-1) + self.epsilon))
                        now_walk.append(nex_point)
                        now_embedding.append(torch.mm(node_embedding[nex_point], self.w_d))
                    else:
                        now_line = od_mat[:][now_walk[-1]]
                        nex_point = numpy.random.choice(a=self.n_nodes, size=1, replace=False,
                                                        p=(now_line.reshape(-1) + self.epsilon) / sum(
                                                            now_line.reshape(-1) + self.epsilon))
                        now_walk.append(nex_point)
                        now_embedding.append(torch.mm(node_embedding[nex_point], self.w_o))

                now_embedding_torch = torch.cat(now_embedding, dim=0)
                walks.append(torch.mean(now_embedding_torch, dim=0))
            walks_torch = torch.stack(walks, dim=0)
            walk_embedding.append(self.aggregator(walks_torch))
        # print(time.time() - x)
        x = time.time()
        walk_embedding_torch = torch.cat(walk_embedding, dim=0)
        # print(time.time() - x)
        # print(walk_embedding_torch.shape)
        return walk_embedding_torch
        '''
