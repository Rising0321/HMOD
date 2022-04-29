import logging
import time
import sys
import argparse
import torch
import operator
import numpy as np
import pickle
from pathlib import Path
import random
from tqdm import trange
import shutil
# torch.autograd.set_detect_anomaly(True)
from model.HierarchicalModel import HierarchicalModel
from utils.utils import EarlyStopMonitor, build_od_matrix, to_device
from utils.data_processing import get_od_data
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error


config = {
    "NYTaxi_sampled": {
        "data_path": "data/NYTaxi/",
        "input_len": 1800,
        "output_len": 1800,
        "day_cycle": 86400,
        "train_day": 5,
        "val_day": 1,
        "test_day": 1,
        "day_start": -1,
        "day_end": 86401,
        "sample": 1,
        "val": 0.75,
        "test": 0.875,
        "start_weekday": 2,
        "n_nodes": 63
    }

}

### Argument and global variables
parser = argparse.ArgumentParser('CTOD training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. NYTaxi_sampled)',
                    default='NYTaxi_sampled')
parser.add_argument('--seed', type=int, default=1, help='Batch_size')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='path of the best model')
parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:3", help='Idx for the gpu to use: cpu, cuda:0, etc.')

parser.add_argument('--loss', type=str, default="odloss", help='Loss function')
parser.add_argument('--message_dim', type=int, default=192, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=288, help='Dimensions of the memory for '
                                                                'each node')
parser.add_argument('--passing_dep', type=int, default=1, help='Number of depth for message')

parser.add_argument('--d_models', type=int, default=4, help='Number of hire-time in model')
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
seed = args.seed
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

NUM_EPOCH = args.n_epoch
device = args.device
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim
passing_dep = args.passing_dep
d_models = args.d_models

input_len = config[DATA]["input_len"]
output_len = config[DATA]["output_len"]
day_cycle = config[DATA]["day_cycle"]
day_start = config[DATA]["day_start"]
day_end = config[DATA]["day_end"]
sample = config[DATA]["sample"]

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.data}-{args.suffix}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.data}-{args.suffix}-{epoch}.pth'
results_path = "results/{}_{}.pkl".format(args.data, args.suffix)
Path("results/").mkdir(parents=True, exist_ok=True)

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(f"log/{str(time.time())}_restore_{args.suffix}.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
n_nodes, node_features, full_data, train_data, val_data, test_data, val_time, test_time, all_time = get_od_data(
    config[DATA])

model = HierarchicalModel(device=device, n_nodes=n_nodes, node_features=node_features,
                          message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                          output=output_len, passing_dep=passing_dep, d_models=d_models)


class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.pro = torch.nn.ReLU()

    def forward(self, predict, truth):
        mask = (truth < 1)
        mask2 = (predict > 0)
        loss = torch.mean(((predict - truth) ** 2) * ~mask + ((mask2 * predict - truth) ** 2) * mask)
        return loss


if args.loss == "odloss":
    logger.info("self od loss!!!!!")
    criterion = OD_loss()
else:
    criterion = torch.nn.MSELoss()
    logger.info("mse loss!!!!!")

model = model.to(device)

val_mses = []
epoch_times = []
total_epoch_times = []
train_losses = []

od_mats = []

heads = []
tail1s = []
tail2s = []
preod_sum = []
real_od_sum = []
train_walks = []

test_flag = 0

val_heads = []
val_tail1s = []
val_tail2s = []
val_ods = []
preod_sum_val = []
test_walks = []

test_heads = []
test_tail1s = []
test_tail2s = []
test_ods = []
preod_sum_test = []
val_walks = []


def Reverse(lst):
    return [ele for ele in reversed(lst)]


def cal_embedding(od_mat):
    epsilon = 1e-5
    repeat_time = 1
    walk_ret1 = []
    walk_ret2 = []
    for i in range(0, n_nodes):
        walks = []
        for repeat in range(repeat_time):
            now_walk = [i]
            for times in range(1 << (repeat + 1)):
                if times & 1:
                    now_line = od_mat[:][now_walk[-1]]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=(now_line.reshape(-1) + epsilon) / sum(
                                                     now_line.reshape(-1) + epsilon))
                    now_walk.append(nex_point[0])

                else:
                    now_line = od_mat[now_walk[-1]][:]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=(now_line.reshape(-1) + epsilon) / sum(
                                                     now_line.reshape(-1) + epsilon))
                    now_walk.append(nex_point[0])
            walks.append(now_walk)
        walk_ret1.append(walks)
        walks = []
        for repeat in range(repeat_time):
            now_walk = [i]
            for times in range(1 << (repeat + 1)):
                if times & 1:
                    now_line = od_mat[now_walk[-1]][:]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=(now_line.reshape(-1) + epsilon) / sum(
                                                     now_line.reshape(-1) + epsilon))
                    now_walk.append(nex_point[0])
                else:
                    now_line = od_mat[:][now_walk[-1]]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=(now_line.reshape(-1) + epsilon) / sum(
                                                     now_line.reshape(-1) + epsilon))
                    now_walk.append(nex_point[0])
            now_walk = Reverse(now_walk)
            walks.append(now_walk)
        walk_ret2.append(walks)
    return [walk_ret1, walk_ret2]


od_tim = [np.zeros([n_nodes, n_nodes])]


def copy(a):
    if isinstance(a, list):
        return [copy(b) for b in a]
    else:
        return a


def cal_prob(now_line, tim):
    import math
    temp = [math.exp((tim - gg) / day_cycle) for gg in now_line]
    sum_temp = sum(temp)
    return [i / sum_temp for i in temp]


def cal_embedding_cont(ed):
    epsilon = 1e-5
    repeat_time = 1
    walk_ret1 = []
    walk_ret2 = []
    for i in range(0, n_nodes):
        walks = []
        for repeat in range(repeat_time):
            now_walk = [i]
            now_pos = [ed // input_len]
            for times in range(1 << (repeat + 1)):
                if times & 1:
                    now_line = od_tim[now_pos[-1]][:][now_walk[-1]]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=cal_prob(now_line, ed))
                    now_walk.append(nex_point[0])
                    now_pos.append(int(od_tim[now_pos[-1]][nex_point[0]][now_walk[-1]] // input_len + 1))

                else:
                    now_line = od_tim[now_pos[-1]][now_walk[-1]][:]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=cal_prob(now_line, ed))
                    now_walk.append(nex_point[0])
                    now_pos.append(int(od_tim[now_pos[-1]][now_walk[-1]][nex_point[0]] // input_len + 1))
            walks.append(now_walk)
        walk_ret1.append(walks)
        walks = []
        for repeat in range(repeat_time):
            now_walk = [i]
            now_pos = [ed // input_len]
            for times in range(1 << (repeat + 1)):
                if times & 1:
                    now_line = od_tim[now_pos[-1]][now_walk[-1]][:]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=cal_prob(now_line, ed))
                    now_walk.append(nex_point[0])
                    now_pos.append(int(od_tim[now_pos[-1]][now_walk[-1]][nex_point[0]] // input_len + 1))
                else:
                    now_line = od_tim[now_pos[-1]][:][now_walk[-1]]
                    nex_point = np.random.choice(a=n_nodes, size=1, replace=False,
                                                 p=cal_prob(now_line, ed))
                    now_walk.append(nex_point[0])
                    now_pos.append(int(od_tim[now_pos[-1]][nex_point[0]][now_walk[-1]] // input_len + 1))
            now_walk = Reverse(now_walk)
            walks.append(now_walk)
        walk_ret2.append(walks)
    return [walk_ret1, walk_ret2]


def eval_od_prediction(model, data, st, ed, device, config, n_nodes):
    input_len = config["input_len"]
    output_len = config["output_len"]
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    sample = config["sample"]
    label, prediction = [], []

    with torch.no_grad():
        model = model.eval()
        num_test_batch = (ed - st - input_len) // output_len

        head = 0
        tail1 = 0
        tail2 = 0
        data_size = len(data.timestamps)
        empty_flag = 1
        ed2 = 0
        for j in tqdm(range(num_test_batch)):
            st1 = j * output_len + st
            ed1 = j * output_len + input_len + st
            ed2 = (j + 1) * output_len + input_len + st

            now = copy(od_tim[-1])
            while head < data_size and data.timestamps[head] < st1:
                head += 1
            while tail1 < data_size and data.timestamps[tail1] < ed1:
                now[data.sources[tail1]][data.destinations[tail1]] = data.timestamps[tail1]
                tail1 += 1
            while tail2 < data_size and data.timestamps[tail2] < ed2:
                tail2 += 1
            od_tim.append(now)
            if test_flag == 0:
                val_heads.append(head)
                val_tail1s.append(tail1)
                val_tail2s.append(tail2)
            else:
                test_heads.append(head)
                test_tail1s.append(tail1)
                test_tail2s.append(tail2)

            od_matrix_real = build_od_matrix(data.sources[tail1:tail2],
                                             data.destinations[tail1:tail2], n_nodes)
            if test_flag == 0:
                val_ods.append(od_matrix_real)
            else:
                test_ods.append(od_matrix_real)

            walks = []
            for _ in range(d_models + 1):
                walks.append([])

            pre_ods = []
            for i in range(0, d_models):
                pre_od = np.zeros(shape=[n_nodes, n_nodes])
                if j - (1 << i) < 0:
                    pre_ods.append(pre_od)
                    walks[i].append(cal_embedding(pre_od))
                else:
                    for k in range(j - (1 << i), j):
                        if test_flag == 0:
                            pre_od += val_ods[k]
                        else:
                            pre_od += test_ods[k]
                    walks[i].append(cal_embedding(pre_od))
                    pre_ods.append(pre_od)

            walks[d_models].append(cal_embedding_cont(ed1))

            if test_flag == 0:
                preod_sum_val.append(pre_ods)
                val_walks.append(walks)
            else:
                preod_sum_test.append(pre_ods)
                test_walks.append(walks)
        now = copy(od_tim[-1])
        while tail1 < data_size and data.timestamps[tail1] < ed2:
            now[data.sources[tail1]][data.destinations[tail1]] = data.timestamps[tail1]
            tail1 += 1
        od_tim.append(now)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
num_batch = (val_time - input_len) // output_len
train_data_size = len(train_data.timestamps)
cal_head_flag = False

head, tail1, tail2 = 0, 0, 0  # [head,tail1) nowtime [tail1,tail2) nowtime+τ

batch_range = trange(num_batch)

ed2 = 0
for j in batch_range:

    ### Training
    st1 = j * output_len
    ed1 = j * output_len + input_len
    ed2 = (j + 1) * output_len + input_len
    now = copy(od_tim[-1])
    while head < train_data_size and train_data.timestamps[head] < st1:
        head += 1
    while tail1 < train_data_size and train_data.timestamps[tail1] < ed1:
        now[train_data.sources[tail1]][train_data.destinations[tail1]] = train_data.timestamps[tail1]
        tail1 += 1
    while tail2 < train_data_size and train_data.timestamps[tail2] < ed2:
        tail2 += 1

    heads.append(head)
    tail1s.append(tail1)
    tail2s.append(tail2)
    od_tim.append(now)

    od_matrix_real = build_od_matrix(train_data.sources[tail1:tail2],
                                     train_data.destinations[tail1:tail2], n_nodes)
    od_mats.append(od_matrix_real)

    pre_ods = []

    walks = []
    for k in range(d_models + 1):
        walks.append([])

    for i in range(0, d_models):
        pre_od = np.zeros(shape=[n_nodes, n_nodes])
        if j - (1 << i) < 0:
            pre_ods.append(pre_od)
            walks[i].append(cal_embedding(pre_od))
        else:
            for k in range(j - (1 << i), j):
                pre_od += od_mats[k]
            pre_ods.append(pre_od)
            walks[i].append(cal_embedding(pre_od))

    walks[d_models].append(cal_embedding_cont(ed1))

    preod_sum.append(pre_ods)
    train_walks.append(walks)

now = copy(od_tim[-1])
while tail1 < train_data_size and train_data.timestamps[tail1] < ed2:
    now[train_data.sources[tail1]][train_data.destinations[tail1]] = train_data.timestamps[tail1]
    tail1 += 1
od_tim.append(now)

cal_head_flag = True
test_flag = 0
### Validation
print("================================Val================================")
eval_od_prediction(model=model,
                   data=val_data,
                   st=val_time,
                   ed=test_time,
                   device=device,
                   config=config[DATA],
                   n_nodes=config[DATA]["n_nodes"])

test_flag = 1
# Test
print("================================Test================================")
eval_od_prediction(model=model,
                   data=test_data,
                   st=test_time,
                   ed=all_time,
                   device=device,
                   config=config[DATA],
                   n_nodes=config[DATA][
                       "n_nodes"])

np.save("processed_data%s.npy" % DATA,
        [heads, tail1s, tail2s, od_mats, preod_sum, train_walks,
         val_heads, val_tail1s, val_tail2s, val_ods, preod_sum_val, val_walks,
         test_heads, test_tail1s, test_tail2s, test_ods, preod_sum_test, test_walks]
        )
