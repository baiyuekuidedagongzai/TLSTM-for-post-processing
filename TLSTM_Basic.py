import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import bb
from sklearn.metrics import mean_squared_error
from openpyxl import load_workbook
import shap

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

excel_file = r"C:\Users\PC\Desktop\北江SWATLSTM\北江.xlsx"
i = 0
nse_list = []
a_sim = np.empty([72, 1])
b_sim = np.empty([72, 1])
c_sim = np.empty([72, 1])

BATCHSIZE = 12
START = 1 - 1  # 起始2003
TRAIN_END = 9 * 12 + 1 - 1
TEST_END = 15 * 12 + 1 - 1
FU_DAY = 0
scenario = [1,0,1,0,0,0,0,0,0,1]

head_q = np.zeros((180, 1))
nse_comb = []
q_data = np.array(pd.read_excel(excel_file, sheet_name='高道'))
q_data_in = q_data[:, 2:]  # (180,30),prec,et,sur,gw,lat,wyld
sub_num = int(q_data_in.shape[1] / (len(scenario) - 1))
q_data_in = np.concatenate((q_data_in, head_q), axis=1)
q_data_in = bb.scenario_datain(q_data_in, sub_num, scenario)
q_data_out = q_data[:, 1].reshape(-1, 1)  # (180,1)

scaler_in = MinMaxScaler(feature_range=(-1, 1))
scaler_out = MinMaxScaler(feature_range=(-1, 1))
train_in = q_data_in[START: TRAIN_END + 1, :]
train_out = q_data_out[START: TRAIN_END + 1].reshape(-1, 1)
train_in_n = scaler_in.fit_transform(train_in)
train_out_n = scaler_out.fit_transform(train_out)

num_layers = 2
hidden_size = 41
dropout = 0.42
windows = 12
l2 = 0
lstm_model = bb.create_model(train_in_n, train_out_n, num_layers, hidden_size, dropout, l2, windows,
                             fu_day=FU_DAY, eporchs=200, train_end=TRAIN_END)
lstm_model.to(device)  # 移动模型到GPU
model = lstm_model.eval()

# 下游训练要用到高道训练集出流
a_train_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=FU_DAY,
                                    verify_end=START + windows - 1, test_end=TRAIN_END)
a_train_out = bb.call(lstm_model, a_train_in_raw.to(device), scaler_out)

a_test_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=FU_DAY,
                                   verify_end=TRAIN_END, test_end=TEST_END)
a_test_out = bb.call(lstm_model, a_test_in_raw.to(device), scaler_out)

tobs = q_data_out[TRAIN_END + FU_DAY: TEST_END]
tnse = bb.NSE_Calculator(a_test_out, tobs)
tkge = bb.KGE_Calculator(a_test_out, tobs)
print('高道tnse=', tnse,'KGE=',tkge)
nse_comb.append(tnse)

gaodao_q = np.concatenate((q_data_out[START: windows - 1], a_train_out, a_test_out), axis=0)


#========================================================================飞来峡=============================================================
q_data = np.array(pd.read_excel(excel_file, sheet_name='飞来峡'))
q_data_in = q_data[:, 2:]  # (180,30),prec,et,sur,gw,lat,wyld
sub_num = int(q_data_in.shape[1] / (len(scenario) - 1))
q_data_in = np.concatenate((q_data_in, gaodao_q), axis=1)
q_data_in = bb.scenario_datain(q_data_in, sub_num, scenario)
q_data_out = q_data[:, 1].reshape(-1, 1)  # (180,1)

scaler_in = MinMaxScaler(feature_range=(-1, 1))
scaler_out = MinMaxScaler(feature_range=(-1, 1))

train_in = q_data_in[START: TRAIN_END + 1, :]
train_out = q_data_out[START: TRAIN_END + 1].reshape(-1, 1)
train_in_n = scaler_in.fit_transform(train_in)
train_out_n = scaler_out.fit_transform(train_out)

num_layers = 2
hidden_size =58
dropout = 0.39
windows = 12
l2 = 0.00
lstm_model = bb.create_model(train_in_n, train_out_n, num_layers, hidden_size, dropout, l2, windows, fu_day=FU_DAY, eporchs=200, train_end=TRAIN_END)
lstm_model.to(device)  # 移动模型到GPU
model = lstm_model.eval()

# 下游训练要用到高道训练集出流
b_train_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=FU_DAY, verify_end=START + windows - 1, test_end=TRAIN_END)
b_train_out = bb.call(lstm_model, b_train_in_raw.to(device), scaler_out)

b_test_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=FU_DAY, verify_end=TRAIN_END, test_end=TEST_END)
b_test_out = bb.call(lstm_model, b_test_in_raw.to(device), scaler_out)

tobs = q_data_out[TRAIN_END + FU_DAY: TEST_END]
tnse = bb.NSE_Calculator(b_test_out, tobs)
tkge = bb.KGE_Calculator(b_test_out, tobs)
print('飞来峡tnse=', tnse,'KGE=',tkge)
nse_comb.append(tnse)

feilaixia_q = np.concatenate((q_data_out[START: windows - 1], b_train_out, b_test_out), axis=0)


#========================================================================三水=============================================================
q_data = np.array(pd.read_excel(excel_file, sheet_name='三水'))
q_data_in = q_data[:, 2:]  # (180,30),prec,et,sur,gw,lat,wyld
sub_num = int(q_data_in.shape[1] / (len(scenario) - 1))
q_data_in = np.concatenate((q_data_in, feilaixia_q), axis=1)
q_data_in = bb.scenario_datain(q_data_in, sub_num, scenario)
q_data_out = q_data[:, 1].reshape(-1, 1)  # (180,1)

scaler_in = MinMaxScaler(feature_range=(-1, 1))
scaler_out = MinMaxScaler(feature_range=(-1, 1))

train_in = q_data_in[START: TRAIN_END + 1, :]
train_out = q_data_out[START: TRAIN_END + 1].reshape(-1, 1)
train_in_n = scaler_in.fit_transform(train_in)
train_out_n = scaler_out.fit_transform(train_out)

num_layers = 2
hidden_size = 40
dropout = 0.18
windows = 12
l2 = 0.01
lstm_model = bb.create_model(train_in_n, train_out_n, num_layers, hidden_size, dropout, l2, windows,
                             fu_day=FU_DAY, eporchs=200, train_end=TRAIN_END)
lstm_model.to(device)  # 移动模型到GPU
model = lstm_model.eval()

# 下游训练要用到高道训练集出流
c_train_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=FU_DAY,
                                    verify_end=START + windows - 1, test_end=TRAIN_END)
c_train_out = bb.call(lstm_model, c_train_in_raw.to(device), scaler_out)

c_test_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=FU_DAY,
                                   verify_end=TRAIN_END, test_end=TEST_END)
c_test_out = bb.call(lstm_model, c_test_in_raw.to(device), scaler_out)

tobs = q_data_out[TRAIN_END + FU_DAY: TEST_END]
tnse = bb.NSE_Calculator(c_test_out, tobs)
tkge = bb.KGE_Calculator(c_test_out, tobs)
print('三水tnse=', tnse,'KGE=',tkge)
nse_comb.append(tnse)