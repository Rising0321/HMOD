import logging
import math
from modules.random_walk_embedding import random_walk_embedding
import numpy as np
import torch
from collections import defaultdict

from model.ContinuousModel import ContinuousModel
from model.DiscreteModel import DiscreteModel
from utils.utils import PredictionLayer, PredictionLayer2, erase_line
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module


class HierarchicalModel(torch.nn.Module):
    def __init__(self, device,
                 n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64,
                 output=30, passing_dep=2, d_models=2, enable_heter=1):
        super(HierarchicalModel, self).__init__()

        self.dmodels = d_models

        self.cmodel = ContinuousModel(device=device, n_nodes=n_nodes, node_features=node_features,
                                      message_dimension=message_dimension, memory_dimension=memory_dimension,
                                      output=output, passing_dep=passing_dep, id=d_models, enable_heter=enable_heter)

        self.dmodel = torch.nn.ModuleList()
        for i in range(d_models):
            self.dmodel.append(DiscreteModel(device=device, n_nodes=n_nodes, node_features=node_features,
                                             message_dimension=message_dimension, memory_dimension=memory_dimension,
                                             output=output, passing_dep=passing_dep, id=i, enable_heter=enable_heter))

        self.predict_od = PredictionLayer(memory_dimension * (d_models + 1), memory_dimension, n_nodes)

    def compute_od_matrix(self, source_nodes, destination_nodes, timestamps_batch_torch,
                          edge_timediff, now_time, begin_time, od_mat, iter, walks,
                          predict_od=True):
        messages = []
        messages.append(self.cmodel.saved_messages)
        # print(self.cmodel.unique_messages)
        # print(self.cmodel.saved_messages)
        for i in range(self.dmodels):
            messages.append(self.dmodel[i].saved_messages)
            # print(self.dmodel[i].saved_messages)
        messages = torch.stack(messages)
        # print(erase_line(messages, 0).transpose(0, 1).transpose(1, 2))
        # print(erase_line(messages, 1).transpose(0, 1).transpose(1, 2))

        embeddings = []
        # 第一维：层数 第二维：点：第三维：不同的层 第四维：embedding
        embeddings.append(self.cmodel.compute_temporal_embedding(
            source_nodes, destination_nodes, timestamps_batch_torch,
            edge_timediff, now_time, begin_time,
            other_message=erase_line(messages, 0).transpose(0, 1).transpose(1, 2), predict_od=predict_od, walks=walks,
            iter=iter))
        self.cmodel.h_update_memory()

        for i in range(self.dmodels):
            embeddings.append(self.dmodel[i].compute_temporal_embedding(
                od_mat[i],
                other_message=erase_line(messages, i + 1).transpose(0, 1).transpose(1, 2), predict_od=predict_od,
                walks=walks, iter=iter))

        for i in range(self.dmodels):
            if iter % (1 << i) == 0:
                self.dmodel[i].h_update_memory()

        if predict_od:
            node_embeddings = torch.cat(embeddings, dim=-1)  # torch.stack(embeddings) / (self.dmodels + 1)
            # node_embeddings = torch.sum(embeddings, dim=0)
            return self.predict_od(node_embeddings)

    def init_memory(self):
        self.cmodel.memory.__init_memory__()
        for i in range(self.dmodels):
            self.dmodel[i].memory.__init_memory__()

    def backup_memory(self):
        backup_mem = [self.cmodel.memory.backup_memory()]

        for i in range(self.dmodels):
            backup_mem.append(self.dmodel[i].memory.backup_memory())
        return backup_mem

    def restore_memory(self, memories):
        self.cmodel.memory.restore_memory(memories[0])
        for i in range(self.dmodels):
            self.dmodel[i].memory.restore_memory(memories[i + 1])

    def detach_memory(self):
        self.cmodel.memory.detach_memory()
        for i in range(self.dmodels):
            self.dmodel[i].detach_memory()

    def get_memory(self):
        return self.memory.memory.data
