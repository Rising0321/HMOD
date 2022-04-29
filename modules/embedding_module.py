from torch import nn
import math


class EmbeddingModule(nn.Module):
    def __init__(self, n_node_features, dropout):
        super(EmbeddingModule, self).__init__()
        self.n_node_features = n_node_features
        self.dropout = dropout

    def compute_embedding(self, memory, nodes, time_diffs=None):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, nodes, time_diffs=None):
        return memory[nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, n_node_features, dropout=0.1):
        super(TimeEmbedding, self).__init__(n_node_features, dropout)

        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features,)

    def compute_embedding(self, memory, nodes, time_diffs=None):
        source_embeddings = memory[nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
        # print(self.embedding_layer.weight.grad)na
        return source_embeddings


def get_embedding_module(module_type, n_node_features, dropout=0.1):
    if module_type == "identity":
        return IdentityEmbedding(n_node_features=n_node_features, dropout=dropout)
    elif module_type == "time":
        return TimeEmbedding(n_node_features=n_node_features, dropout=dropout)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
