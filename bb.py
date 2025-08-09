import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import Selfornot_LSTM
import openpyxl
from permetrics.regression import RegressionMetric

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def NSE_Calculator(sim_seq, obs_seq):
    avg = np.mean(obs_seq)
    numerator = np.sum((sim_seq - obs_seq) ** 2)
    denominator = np.sum((obs_seq - np.full(obs_seq.shape, avg)) ** 2)
    NSE = 1 - numerator / denominator
    return NSE

def KGE_Calculator(sim, obs):
    evaluator = RegressionMetric(obs, sim)
    KGE = evaluator.kling_gupta_efficiency()
    return KGE

def R2_Calculator(sim, obs):
    sim = np.array(sim)
    obs = np.array(obs)
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    numerator = np.sum((obs - mean_obs)*(sim - mean_sim))
    denominator = np.sqrt(np.sum((obs - mean_obs)**2) * np.sum((sim - mean_sim)**2))
    R2 = (numerator / denominator)**2
    return R2

def RMSE_Calculator(sim, obs):
    sim = np.array(sim)
    obs = np.array(obs)
    RMSE = np.sqrt(((sim - obs)**2).mean())
    return RMSE

def create_model(input_data_n, output_data_n, num_layers, hidden_size, dropout, l2, windows, fu_day=0, eporchs=200,
                 train_end=108, batchsize=12):
    in_d = input_data_n.shape[1]
    train_in = input_data_n[:len(input_data_n) - fu_day]
    train_out = output_data_n[fu_day:]
    ts_train_in = torch.tensor(train_in, dtype=torch.float32).to(device)
    ts_train_out = torch.tensor(train_out, dtype=torch.float32).to(device)

    batch_train_in = list()
    batch_train_out = list()
    for i in range(len(ts_train_in) - windows + 1):
        batch_train_in.append(ts_train_in[i: i + windows])
        batch_train_out.append(ts_train_out[i: i + windows])

    batch_train_in = pad_sequence(batch_train_in).to(device)
    batch_train_out = pad_sequence(batch_train_out).to(device)

    total_batches = batch_train_in.shape[1] // batchsize

    model = Selfornot_LSTM.AttentionLSTM(in_d, hidden_size, num_layers, dropout, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=l2)

    for i in range(eporchs):
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batchsize
            end_idx = (batch_idx + 1) * batchsize
            optimizer.zero_grad()
            outputs = model(batch_train_in[:, start_idx:end_idx])
            loss = criterion(outputs, batch_train_out[:, start_idx:end_idx])
            loss.backward()
            optimizer.step()

    return model


def create_test_raw(input_data, output_data, scaler_in, scaler_out, windows, fu_day=0, verify_end=144, test_end=180):
    temp_test_in = input_data[verify_end + 1 - windows: test_end - fu_day, :]
    temp_test_in_n = scaler_in.transform(temp_test_in)

    test_in = list()
    for i in range(len(temp_test_in_n) - windows + 1):
        test_in_part = torch.tensor(temp_test_in_n[i: i + windows, :], dtype=torch.float32).to(device)
        test_in.append(test_in_part)

    batch_test_in = pad_sequence(test_in).to(device)
    return batch_test_in


def call(lstm_model, in_raw, scaler_out):
    model = lstm_model.eval()
    out_n = list()
    with torch.no_grad():
        for b in range(in_raw.shape[1]):
            temp_pred_n = model(in_raw[:, b, :].unsqueeze(1)).cpu().numpy().reshape(-1, 1)
            out_n.append(temp_pred_n[-1])

    out = scaler_out.inverse_transform(np.array(out_n))
    return out


def scenario_datain(train_in, sub_num, scenario_list = [1,1,1,1,1,1,1]):
    temp = np.empty((train_in.shape[0], 0))
    for i in range(len(scenario_list)):
        if i != len(scenario_list) - 1 and scenario_list[i] == 1:
            temp = np.concatenate((temp, train_in[:, sub_num * i: sub_num * (i + 1)]), axis = 1)
        elif i == len(scenario_list) - 1 and scenario_list[i] == 1:
            temp = np.concatenate((temp, train_in[:, sub_num * i, np.newaxis]), axis = 1)
    return temp