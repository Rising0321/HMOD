import logging
import math

import numpy as np
import torch
from collections import defaultdict

from modules.random_walk_embedding import random_walk_embedding
from utils.utils import PredictionLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module


class DiscreteModel(torch.nn.Module):
    def __init__(self, device, id,
                 n_nodes=268, node_features=None,
                 message_dimension=64, memory_dimension=64,
                 output=30, passing_dep=2, enable_heter=1):
        super(DiscreteModel, self).__init__()
        dropout = 0.1
        self.id = id
        self.output = output
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.n_node_features = memory_dimension + n_nodes + message_dimension * enable_heter
        self.n_nodes = n_nodes
        self.n_regions = math.ceil(math.sqrt(self.n_nodes))
        self.n_graphs = 1

        self.memory_dimension = memory_dimension
        raw_message_dimension = self.n_node_features
        # print(raw_message_dimension)
        self.memory = Memory(n_nodes=self.n_nodes,
                             memory_dimension=self.memory_dimension,
                             input_dimension=message_dimension,
                             message_dimension=message_dimension,
                             device=device)

        self.message_aggregator = get_message_aggregator(aggregator_type="sum", device=device)
        self.message_function = get_message_function(module_type="mlp",
                                                     raw_message_dimension=raw_message_dimension,
                                                     message_dimension=message_dimension)
        self.memory_updater = get_memory_updater(module_type="gru",  # gru rnn
                                                 memory=self.memory,
                                                 message_dimension=message_dimension,
                                                 memory_dimension=self.memory_dimension,
                                                 device=device,
                                                 passing_dep=passing_dep)

        self.iden_embedding_module = get_embedding_module(module_type="identity",
                                                          n_node_features=self.memory_dimension,
                                                          dropout=dropout)
        self.time_embedding_module = get_embedding_module(module_type="identity",
                                                          n_node_features=self.memory_dimension,
                                                          dropout=dropout)

        self.message_demention = message_dimension

        self.predict_od = PredictionLayer(self.memory_dimension, self.memory_dimension, self.n_nodes)

        self.passing_dep = passing_dep

        self.saved_messages = torch.zeros([self.passing_dep + 1, self.n_nodes, message_dimension]).to(device)

        self.random_walk_embeddong =  random_walk_embedding(n_nodes=n_nodes,
                                                           input_dimension=memory_dimension,
                                                           output_dimension=message_dimension,
                                                           repeat_time=1,
                                                           device=device)

        self.enable_heter = enable_heter

    def compute_od_matrix(self, od_mat, other_message=None, predict_od=True):
        print(self.enable_heter)
        destination_nodes = torch.arange(self.n_nodes)

        destination_node_embedding = self.iden_embedding_module.compute_embedding(memory=self.memory.memory.data,
                                                                                  nodes=destination_nodes)

        # Compute node_level messages
        source_nodes = destination_nodes
        source_messages = torch.cat([destination_node_embedding, torch.FloatTensor(od_mat).to(self.device)], dim=-1)
        '''
        unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                      edge_timediff,
                                                                      destination_node_embedding,
                                                                      timestamps_batch_torch)
        '''
        source_messages = self.message_function.compute_message(source_messages)
        updated_memory, updated_last_update, unique_nodes, unique_messages, unique_timestamps, unique_raw_messages = self.get_updated_memory(
            source_nodes, source_messages)

        # 3. Compute multi-scale messages

        od_matrix = None
        node_embeddings = None
        if predict_od:
            # 4. Get updated memories and updated embeddings
            recent_node_embeddings = self.time_embedding_module.compute_embedding(memory=updated_memory,
                                                                                  nodes=list(range(self.n_nodes)))

            # 5. Fuse multi-scale embeddings for prediction
            node_node_embeddings = torch.cat([recent_node_embeddings], dim=1)
            node_embeddings = torch.cat([node_node_embeddings], dim=1)
            od_matrix = self.predict_od(node_embeddings)

        # 6. Save memories and messaget for next prediction
        saved_memories = [self.memory.memory.data.cpu().numpy()]
        saved_messages = torch.zeros([self.n_nodes, unique_raw_messages.shape[1]])
        saved_messages[unique_nodes] = unique_raw_messages.cpu().detach()
        self.update_memory(unique_nodes, unique_messages, unique_timestamps)
        return od_matrix, saved_messages.numpy(), saved_memories

    def compute_temporal_embedding(self, od_mat, walks, iter, other_message=None, predict_od=True):

        destination_nodes = torch.arange(self.n_nodes)

        destination_node_embedding = self.iden_embedding_module.compute_embedding(memory=self.memory.memory.data,
                                                                                  nodes=destination_nodes)

        # Compute node_level messages
        source_nodes = destination_nodes
        if self.enable_heter:
            source_messages = torch.cat([destination_node_embedding,
                                         torch.FloatTensor(od_mat).to(self.device),
                                         self.random_walk_embeddong.cal_embedding_processed(walks[iter][self.id][0],
                                                                                            self.memory.memory.data)],
                                        dim=-1)
        else:
            source_messages = torch.cat([destination_node_embedding,
                                         torch.FloatTensor(od_mat).to(self.device)
                                         ],
                                        dim=-1)
        unique_raw_messages = self.message_function.compute_message(source_messages)
        # print(self.message_function.mlp[0].weight)

        updated_memory, updated_last_update, unique_nodes, unique_messages, unique_timestamps, unique_raw_messages = self.get_updated_memory(
            source_nodes, unique_raw_messages, other_message)
        # print(self.memory_updater.fc1s[1].weight)
        # 3. Compute multi-scale messages

        node_embeddings = None
        if predict_od:
            # 4. Get updated memories and updated embeddings
            node_embeddings = self.time_embedding_module.compute_embedding(memory=updated_memory,
                                                                                  nodes=list(range(self.n_nodes)))

            # od_matrix = self.predict_od(node_embeddings)

        # 6. Save memories and messaget for next prediction
        self.unique_nodes = unique_nodes
        self.unique_messages = unique_messages
        self.unique_timestamps = unique_timestamps
        self.other_message = other_message

        # self.update_memory(unique_nodes, unique_messages, unique_timestamps, other_message)
        return node_embeddings

    def h_update_memory(self):
        self.update_memory(self.unique_nodes, self.unique_messages, self.unique_timestamps, self.other_message)

    def update_memory(self, unique_nodes, unique_messages, unique_timestamps, other_message=None):
        self.saved_messages[0][unique_nodes] = unique_messages.detach()
        agg_messages = self.memory_updater.update_memory(unique_nodes, unique_messages, other_message,
                                                         timestamps=unique_timestamps)
        if agg_messages != None:
            for i in range(self.passing_dep):
                self.saved_messages[i + 1][unique_nodes] = agg_messages[i].detach()

    def get_updated_memory(self, unique_nodes, unique_raw_messages, other_message=None):
        # Aggregate messages for the same nodes
        '''
        unique_nodes, unique_raw_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)
        '''
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_raw_messages,
                                                                                     other_message,
                                                                                     timestamps=None)
        return updated_memory, updated_last_update, unique_nodes, unique_raw_messages, None, unique_raw_messages

    def get_raw_messages(self, source_nodes, edge_timediff, destination_node_embedding, edge_times):
        source_message = torch.cat(
            [destination_node_embedding,
             destination_node_embedding * torch.exp(edge_timediff / self.output).unsqueeze(1)],
            dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def init_memory(self):
        self.memory.__init_memory__()

    def backup_memory(self):
        return [self.memory.backup_memory()]

    def restore_memory(self, memories):
        self.memory.restore_memory(memories[0])

    def detach_memory(self):
        self.memory.detach_memory()

    def get_memory(self):
        return self.memory.memory.data
