import logging
import time
import sys
import argparse
import torch
import operator
import numpy as np
from pathlib import Path
import random
from tqdm import trange
import shutil
from model.HierarchicalModel import HierarchicalModel
from utils.utils import EarlyStopMonitor, build_od_matrix, to_device, split_walk
from utils.data_processing import get_od_data, get_preprocessed_data
from tqdm import tqdm
from modules.OD_loss import OD_loss
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


def init(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def init_param():
    ### Argument and global variables
    parser = argparse.ArgumentParser('CTOD training')
    parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. NYTaxi_sampled)',
                        default='NYTaxi_sampled')
    parser.add_argument('--seed', type=int, default=1, help='Batch_size')
    parser.add_argument('--suffix', type=str, default='ny1', help='Suffix to name the checkpoints')
    parser.add_argument('--best', type=str, default='', help='path of the best model')
    parser.add_argument('--n_epoch', type=int, default=500, help='Number of epochs')
    parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default="cuda:1", help='Idx for the gpu to use: cpu, cuda:0, etc.')
    parser.add_argument('--loss', type=str, default="odloss", help='Loss function')
    parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
    parser.add_argument('--passing_dep', type=int, default=1, help='Number of depth for message')
    parser.add_argument('--d_models', type=int, default=4, help='Number of hire-time in model')
    parser.add_argument('--enable_heter', type=int, default=1, help='Whether use OD embedding')
    parser.add_argument('--begin_epoch', type=int, default=-1, help='load or train directly')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


def parse_param(args):
    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.message_dim
    passing_dep = args.passing_dep
    d_models = args.d_models
    enable_heter = args.enable_heter
    begin_epoch = args.begin_epoch

    input_len = config[DATA]["input_len"]
    output_len = config[DATA]["output_len"]
    day_cycle = config[DATA]["day_cycle"]
    day_start = config[DATA]["day_start"]
    day_end = config[DATA]["day_end"]
    sample = config[DATA]["sample"]

    return NUM_EPOCH, device, DATA, NUM_LAYER, LEARNING_RATE, MESSAGE_DIM, MEMORY_DIM, passing_dep, d_models, \
           input_len, output_len, day_cycle, day_start, day_end, sample, enable_heter, begin_epoch


def init_log(args):
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
    return logger, fh, ch, get_checkpoint_path, MODEL_SAVE_PATH


def get_loss(args):
    if args.loss == "odloss":
        logger.info("self od loss!!!!!")
        criterion = OD_loss()
    else:
        criterion = torch.nn.MSELoss()
        logger.info("mse loss!!!!!")
    return criterion


def eval_od_prediction(model, data, st, ed, device, config, n_nodes, test_flag, walks):
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

        for j in tqdm(range(num_test_batch)):
            st1 = j * output_len + st
            ed1 = j * output_len + input_len + st
            ed2 = (j + 1) * output_len + input_len + st

            if test_flag == 0:
                head = val_heads[j]
                tail1 = val_tail1s[j]
                tail2 = val_tail2s[j]
            else:
                head = test_heads[j]
                tail1 = test_tail1s[j]
                tail2 = test_tail2s[j]
            if test_flag == 0:
                od_matrix_real = val_ods[j]
                pre_ods = preod_sum_val[j]
            else:
                od_matrix_real = test_ods[j]
                pre_ods = preod_sum_test[j]

            if ed1 % day_cycle < day_start or ed1 % day_cycle > day_end:
                continue

            if head == tail1 or tail1 == tail2:
                continue

            now_time = ed1
            begin_time = st1
            if ed1 % day_cycle >= day_end:
                predict_od = False
            else:
                predict_od = True
            sources_batch, destinations_batch = data.sources[head:tail1:sample], \
                                                data.destinations[head:tail1:sample]
            timestamps_batch = data.timestamps[head:tail1:sample]
            time_diffs_batch = data.timestamps[head:tail1:sample] - now_time
            timestamps_batch_torch = torch.Tensor(timestamps_batch).to(device)
            time_diffs_batch_torch = torch.Tensor(time_diffs_batch).to(device)
            # predict_od = True
            od_matrix_predicted = model.compute_od_matrix(
                sources_batch, destinations_batch, timestamps_batch_torch,
                time_diffs_batch_torch, now_time, begin_time,
                pre_ods, iter=j,
                predict_od=predict_od, walks=walks)
            if predict_od:
                prediction.append(od_matrix_predicted.cpu().numpy())
                label.append(od_matrix_real)

        stacked_prediction = np.stack(prediction)
        stacked_prediction[stacked_prediction < 0] = 0
        stacked_label = np.stack(label)
        reshaped_prediction = stacked_prediction.reshape(-1)
        reshaped_label = stacked_label.reshape(-1)
        mse = mean_squared_error(reshaped_prediction, reshaped_label)
        mae = mean_absolute_error(reshaped_prediction, reshaped_label)
        pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
        smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
                np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
        print(mse, mae, pcc, smape)

    return mse, mae, pcc, smape, stacked_prediction, stacked_label


if __name__ == '__main__':
    args = init_param()
    init(args.seed)
    logger, fh, ch, get_checkpoint_path, MODEL_SAVE_PATH = init_log(args)

    NUM_EPOCH, device, DATA, NUM_LAYER, LEARNING_RATE, MESSAGE_DIM, MEMORY_DIM, passing_dep, d_models, \
    input_len, output_len, day_cycle, day_start, day_end, sample, enable_heter, begin_epoch = parse_param(args)

    ### Extract data for training, validation and testing
    n_nodes, node_features, full_data, train_data, val_data, test_data, val_time, test_time, all_time = get_od_data(
        config[DATA])

    heads, tail1s, tail2s, od_mats, preod_sum, train_walks, \
    val_heads, val_tail1s, val_tail2s, val_ods, preod_sum_val, val_walks, \
    test_heads, test_tail1s, test_tail2s, test_ods, preod_sum_test, test_walks = get_preprocessed_data(DATA)

    model = HierarchicalModel(device=device, n_nodes=n_nodes, node_features=node_features,
                              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                              output=output_len, passing_dep=passing_dep, d_models=d_models, enable_heter=enable_heter)
    model = model.to(device)
    if begin_epoch != -1:
        model.load_state_dict(torch.load(get_checkpoint_path(begin_epoch))["statedict"])
    criterion = get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)

    epoch_times = []
    total_epoch_times = []
    train_losses = []

    num_batch = (val_time - input_len) // output_len
    train_data_size = len(train_data.timestamps)

    for epoch in range(begin_epoch + 1, NUM_EPOCH):
        print("================================Epoch: %d================================" % epoch)
        start_epoch = time.time()
        logger.info('start {} epoch'.format(epoch))

        model.init_memory()

        head, tail1, tail2 = 0, 0, 0  # [head,tail1) nowtime [tail1,tail2) nowtime+τ
        m_loss = []

        model = model.train()
        batch_range = trange(num_batch)
        for j in batch_range:

            ### Training
            st1 = j * output_len
            ed1 = j * output_len + input_len
            ed2 = (j + 1) * output_len + input_len

            head = heads[j]
            tail1 = tail1s[j]
            tail2 = tail2s[j]
            od_matrix_real = od_mats[j]
            pre_ods = preod_sum[j]

            if head == tail1:
                continue

            if ed1 % day_cycle < day_start or ed1 % day_cycle > day_end:
                continue

            time_of_matrix = ed1 % day_cycle // output_len
            day_of_matrix = ed1 // day_cycle
            weekday_of_matrix = ed1 // day_cycle % 7
            time_of_matrix2 = (ed1 + output_len) % day_cycle // output_len
            weekday_of_matrix2 = (ed1 + output_len) // day_cycle % 7

            optimizer.zero_grad()
            start_idx = head
            end_idx = tail1
            now_time = ed1
            begin_time = st1

            if ed1 % day_cycle >= day_end:
                predict_od = False
            else:
                predict_od = True

            sources_batch, destinations_batch = train_data.sources[start_idx:end_idx:sample], \
                                                train_data.destinations[start_idx:end_idx:sample]
            edge_idxs_batch = train_data.edge_idxs[start_idx:end_idx:sample]
            timestamps_batch = train_data.timestamps[start_idx:end_idx:sample]
            time_diffs_batch = train_data.timestamps[start_idx:end_idx:sample] - now_time
            timestamps_batch_torch = torch.Tensor(timestamps_batch).to(device)
            time_diffs_batch_torch = torch.Tensor(time_diffs_batch).to(device)

            # Predict OD, get updated memories and messages
            od_matrix_predicted = model.compute_od_matrix(
                sources_batch, destinations_batch, timestamps_batch_torch,
                time_diffs_batch_torch, now_time, begin_time,
                pre_ods, iter=j, predict_od=predict_od, walks=train_walks)

            #  print(od_matrix_predicted)

            if predict_od:
                # print(od_matrix_predicted)
                loss = criterion(od_matrix_predicted, torch.FloatTensor(od_matrix_real).to(device))
                loss.backward(retain_graph=True)
                optimizer.step()
                m_loss.append(loss.item())
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            batch_range.set_description(f"train_loss: {m_loss[-1]};")

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        ### Validation
        print("================================Val================================")
        val_mse, val_mae, val_pcc, val_smape, _, _ = eval_od_prediction(model=model,
                                                                        data=val_data,
                                                                        st=val_time,
                                                                        ed=test_time,
                                                                        device=device,
                                                                        config=config[DATA],
                                                                        n_nodes=config[DATA]["n_nodes"],
                                                                        test_flag=0,
                                                                        walks=val_walks)

        # Test
        print("================================Test================================")
        test_mse, test_mae, test_pcc, test_smape, prediction, label = eval_od_prediction(model=model,
                                                                                         data=test_data,
                                                                                         st=test_time,
                                                                                         ed=all_time,
                                                                                         device=device,
                                                                                         config=config[DATA],
                                                                                         n_nodes=config[DATA][
                                                                                             "n_nodes"],
                                                                                         test_flag=1,
                                                                                         walks=test_walks)

        # Save temporary results
        train_losses.append(np.mean(m_loss))
        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
        logger.info(
            f'Epoch val metric: mae, mse, rmse, pcc, smape, {val_mae}, {val_mse}, {np.sqrt(val_mse)}, {val_pcc}, {val_smape}')
        logger.info(
            'Test statistics:-- mae: {}, mse: {}, rmse: {}, pcc: {}, smape:{}'.format(test_mae, test_mse,
                                                                                      np.sqrt(test_mse),
                                                                                      test_pcc, test_smape))

        # Early stopping
        ifstop, ifimprove = early_stopper.early_stop_check(val_mse)
        if ifstop:
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        else:
            torch.save(
                {"statedict": model.state_dict(), "memory": model.backup_memory()},
                get_checkpoint_path(epoch))

    logger.info('Saving DyOD model')
    shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
    logger.info('DyOD model saved')
