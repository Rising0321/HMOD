import numpy as np


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, n_nodes):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.n_interactions = len(sources)
        self.unique_nodes = set(list(range(n_nodes)))
        self.n_unique_nodes = n_nodes


def upper_bound(nums, target):
    l, r = 0, len(nums) - 1
    pos = -1
    while l <= r:
        mid = int((l + r) / 2)
        if nums[mid] > target:
            r = mid - 1
            pos = mid
        else:  # >
            l = mid + 1
    return pos


def get_od_data(config):
    whole_data = np.load(config["data_path"] + "sampled.npy").astype("int").reshape([-1, 3])
    print("data loaded")
    all_time = (config["train_day"] + config["val_day"] + config["test_day"]) * config["day_cycle"]
    val_time, test_time = (config["train_day"]) * config["day_cycle"], (config["train_day"] + config["val_day"]) * \
                          config["day_cycle"]
    sources = whole_data[:, 0]
    destinations = whole_data[:, 1]
    timestamps = whole_data[:, 2]
    edge_idxs = np.arange(whole_data.shape[0])
    n_nodes = config["n_nodes"]
    node_features = np.diag(np.ones(n_nodes))

    train_mask = upper_bound(timestamps, val_time)
    val_mask = upper_bound(timestamps, test_time)
    full_data = Data(sources, destinations, timestamps, edge_idxs, n_nodes)
    train_data = Data(sources[:train_mask], destinations[:train_mask], timestamps[:train_mask], edge_idxs[:train_mask],
                      n_nodes)
    val_data = Data(sources[train_mask:val_mask], destinations[train_mask:val_mask], timestamps[train_mask:val_mask],
                    edge_idxs[train_mask:val_mask], n_nodes)
    test_data = Data(sources[val_mask:], destinations[val_mask:], timestamps[val_mask:], edge_idxs[val_mask:], n_nodes)

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                                 full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return n_nodes, node_features, full_data, train_data, val_data, test_data, val_time, test_time, all_time


def get_preprocessed_data(config):
    return np.load("processed_data%s.npy" % config, allow_pickle=True)
