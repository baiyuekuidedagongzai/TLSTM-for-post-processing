import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
import Selfornot_LSTM
import bb

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXCEL_FILE = r"C:\Users\PC\Desktop\北江SWATLSTM\北江.xlsx"
BATCHSIZE = 12
START = 1 - 1  # 起始2003
TRAIN_END = 9 * 12 + 1 - 1
TEST_END = 15 * 12 + 1 - 1
FU_DAY = 0
K = 5

# =====================================================bb.create_model的变体，专门指定成三段方便交叉验证，每次输出指定为验证集部分,且输入数据是已经处理过的
def create_kfold_model(input_data_n, output_data_n, num_layers, hidden_size, dropout, l2, windows, fu_day=0, eporchs=200,
                 tn_start1 = 0, tn_end1 = 0, vf_start2 = 0, vf_end2 = 21, tn_start3 = 21, tn_end3 = 108, batchsize=12):
    in_d = input_data_n.shape[1]
    train_in = input_data_n[:len(input_data_n) - fu_day]
    train_out = output_data_n[fu_day:]
    ts_train_in = torch.tensor(train_in, dtype=torch.float32).to(device)
    ts_train_out = torch.tensor(train_out, dtype=torch.float32).to(device)

    batch_train_in = list()
    batch_train_out = list()
    verify_in_n = list()
    if tn_end1 > tn_start1:
        for i in range(tn_start1, tn_end1 - windows + 1):
            batch_train_in.append(ts_train_in[i: i + windows])
            batch_train_out.append(ts_train_out[i: i + windows])

    for i in range(max(vf_start2 - windows + 1,0), vf_end2 - windows + 1):
        verify_in_n.append(ts_train_in[i: i + windows])

    if tn_end3 > tn_start3:
        for i in range(tn_start3 - windows + 1, tn_end3 - windows + 1):
            batch_train_in.append(ts_train_in[i: i + windows])
            batch_train_out.append(ts_train_out[i: i + windows])
    batch_train_in = pad_sequence(batch_train_in).to(device)
    batch_train_out = pad_sequence(batch_train_out).to(device)
    verify_in_n = pad_sequence(verify_in_n).to(device)
    # print(verify_in_n.shape)

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

    model = model.eval()
    verify_out_n = list()
    with torch.no_grad():
        for b in range(verify_in_n.shape[1]):
            temp_pred_n = model(verify_in_n[:, b, :].unsqueeze(1)).cpu().numpy().reshape(-1, 1)
            verify_out_n.append(temp_pred_n[-1])
    verify_out_n = np.array(verify_out_n)
    return verify_out_n

#==============================================================================================堆叠验证集进行检查
def kfold_vnse(n1, h1, d1, l2_1, w1, head_q, station, scenario_list = [1,1,1,1,1,1,1,1,1,1], excel_file=EXCEL_FILE, batchsize=BATCHSIZE,
           start=START, train_end=TRAIN_END, k = K, fu_day=FU_DAY):

    q_data = np.array(pd.read_excel(excel_file, sheet_name=station))
    q_data_in = q_data[:, 2:]  # (180,30)
    # 上游出流只有一列，作为例外
    sub_num = int(q_data_in.shape[1] / (len(scenario_list) - 1))
    q_data_in = np.concatenate((q_data_in, head_q), axis=1)
    q_data_in = bb.scenario_datain(q_data_in, sub_num, scenario_list)
    q_data_out = q_data[:, 1].reshape(-1, 1)  # (180,1)

    scaler_in = MinMaxScaler(feature_range=(-1, 1))
    scaler_out = MinMaxScaler(feature_range=(-1, 1))
    train_in = q_data_in[start: train_end + 1, :]
    train_out = q_data_out[start: train_end + 1].reshape(-1, 1)
    train_in_n = scaler_in.fit_transform(train_in)
    train_out_n = scaler_out.fit_transform(train_out)

    verify_len = int((train_end - start) / k)
    gaodao_verify = np.empty((0, 1))
    for i in range(k):
        if i < k - 1:
            tn_start1 = 0
            tn_end1 = i * verify_len
            vf_start2 = i * verify_len
            vf_end2 = (i + 1) * verify_len
            tn_start3 = (i + 1) * verify_len
            tn_end3 = train_end
        else:
            tn_start1 = 0
            tn_end1 = i * verify_len
            vf_start2 = i * verify_len
            vf_end2 = train_end
            tn_start3 = train_end
            tn_end3 = train_end

        gaodao_v_n = create_kfold_model(train_in_n, train_out_n, n1, h1, d1, l2_1, w1, fu_day=0, eporchs=200,
                 tn_start1 = tn_start1, tn_end1 = tn_end1, vf_start2 = vf_start2, vf_end2 = vf_end2, tn_start3 = tn_start3, tn_end3 = tn_end3, batchsize=batchsize)
        gaodao_verify = np.concatenate((gaodao_verify, gaodao_v_n),axis = 0)
    gaodao_verify = scaler_out.inverse_transform(gaodao_verify)
    vobs = q_data_out[start + w1 - 1 + fu_day: train_end]
    vnse = bb.NSE_Calculator(gaodao_verify, vobs)
    return vnse

def station(n1, h1, d1, l2_1, w1, head_q, station, scenario_list = [1,1,1,1,1,1,1,1,1,1], excel_file=EXCEL_FILE, batchsize=BATCHSIZE,
           start=START, train_end=TRAIN_END, test_end=TEST_END, fu_day=FU_DAY):
    q_data = np.array(pd.read_excel(excel_file, sheet_name=station))
    q_data_in = q_data[:, 2:]  # (180,30)
    sub_num = int(q_data_in.shape[1] / (len(scenario_list) - 1))
    q_data_in = np.concatenate((q_data_in, head_q), axis=1)
    q_data_in = bb.scenario_datain(q_data_in, sub_num, scenario_list)
    q_data_out = q_data[:, 1].reshape(-1, 1)  # (180,1)

    scaler_in = MinMaxScaler(feature_range=(-1, 1))
    scaler_out = MinMaxScaler(feature_range=(-1, 1))
    train_in = q_data_in[start: train_end + 1, :]
    train_out = q_data_out[start: train_end + 1].reshape(-1, 1)
    train_in_n = scaler_in.fit_transform(train_in)
    train_out_n = scaler_out.fit_transform(train_out)

    num_layers = n1
    hidden_size = h1
    dropout = d1
    windows = w1
    l2 = l2_1
    lstm_model = bb.create_model(train_in_n, train_out_n, num_layers, hidden_size, dropout, l2, windows, fu_day=fu_day,
                                 eporchs=200, train_end=train_end)

    lstm_model = lstm_model.to(device)  # 将模型移动到 GPU
    lstm_model.eval()

    train_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=fu_day,
                                        verify_end=start + windows - 1, test_end=train_end)
    train_out = bb.call(lstm_model, train_in_raw.to(device), scaler_out)  # 移动输入到 GPU

    test_in_raw = bb.create_test_raw(q_data_in, q_data_out, scaler_in, scaler_out, windows, fu_day=fu_day,
                                       verify_end=train_end, test_end=test_end)
    test_out = bb.call(lstm_model, test_in_raw.to(device), scaler_out)  # 移动输入到 GPU

    tobs = q_data_out[train_end + fu_day: test_end]
    tnse = bb.NSE_Calculator(test_out, tobs)
    # print('tnse=', tnse)
    out_q = np.concatenate((q_data_out[start: windows - 1], train_out, test_out), axis=0)
    #out_q = np.concatenate((q_data_out[start: train_end], test_out), axis=0)#训练集以上游观测为输入
    return test_out, out_q

def kfold_tlstm(n1,h1,d1,l21,w1,n2,h2,d2,l22,w2,n3,h3,d3,l23,w3,excel_file=EXCEL_FILE, batchsize=BATCHSIZE,
           start=START, train_end=TRAIN_END, test_end=TEST_END, fu_day=FU_DAY,stations = ['高道','飞来峡','三水'],scenario_list = [1,1,1,1,1,1,1,1,1,1]):
    head_q = np.zeros((180, 1))
    gaodao_test, gaodao_q = station(n1,h1,d1,l21,w1, head_q,  stations[0], scenario_list,excel_file=excel_file, batchsize=batchsize,
           start=start, train_end=train_end, test_end=test_end, fu_day=fu_day)
    feilaixia_test, feilaixia_q = station(n2, h2, d2,l22,w2, gaodao_q,  stations[1],scenario_list, excel_file=excel_file, batchsize=batchsize,
           start=start, train_end=train_end, test_end=test_end, fu_day=fu_day)
    sanshui_test, sanshui_q = station(n3, h3, d3, l23, w3, feilaixia_q, stations[2],scenario_list, excel_file=excel_file, batchsize=batchsize,
           start=start, train_end=train_end, test_end=test_end, fu_day=fu_day)
    return gaodao_test, feilaixia_test, sanshui_test