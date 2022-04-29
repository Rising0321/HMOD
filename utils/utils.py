import numpy as np
import torch


class PredictionLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, n_nodes):
        super().__init__()
        self.n_nodes = n_nodes
        self.fc1 = torch.nn.Linear(dim1, dim2)
        self.fc2 = torch.nn.Linear(dim2, self.n_nodes)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        # print(self.fc2.weight)
        h = self.fc2(self.act(self.fc1(x)))
        return h.reshape([self.n_nodes, self.n_nodes])


class PredictionLayer2(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc = torch.nn.Linear(dim, dim)
        self.act = torch.nn.ReLU()
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # print(self.fc1.weight.grad)
        h = self.act(self.fc(x))
        return torch.mm(x, h.t())


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=False, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round, (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance


def build_od_matrix(source, destination, n_nodes):
    od_matrix = np.zeros(shape=[n_nodes, n_nodes])
    mylen = len(source)
    for i in range(mylen):
        od_matrix[int(source[i])][int(destination[i])] += 1
    return od_matrix


def to_device(data, device):
    if isinstance(data, list):
        return [to_device(d, device) for d in data]
    elif isinstance(data, np.ndarray):
        return torch.Tensor(data).to(device)
    elif isinstance(data, dict):
        return {k: to_device(data[k], device) for k in data.keys()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return torch.Tensor(data).to(device)


def erase_line(data, idx):
    return data[torch.arange(data.size(0)) != idx]


def split_walk(walks, device):
    walk_o = np.array(walks)[:, :, :, 0::2]  # t * n * walk_num * len
    walk_d = np.array(walks)[:, :, :, 1::2]  # t * n * walk_num * len
    return [torch.FloatTensor(walk_o).to(device), torch.FloatTensor(walk_d).to(device)]
