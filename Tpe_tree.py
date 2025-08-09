import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import GridSearchCV
import gc
import kfold_tlstm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCEL_FILE = r"C:\Users\PC\Desktop\北江SWATLSTM\北江.xlsx"
# 做对照TVAR不变；非树\非树VAR改scenario
BATCHSIZE = 12
START = 1 - 1  # 起始2003
TRAIN_END = 9 * 12 + 1 - 1
TEST_END = 15 * 12 + 1 - 1
K = 5
FU_DAY = 0
scenarios = pd.read_excel(EXCEL_FILE, sheet_name='情境表', index_col=0)
for index, scenario in scenarios.iterrows():
    output_path = fr'C:\Users\PC\Desktop\北江SWATLSTM\结果\{index}.xlsx'
    scenario = scenario.tolist()
    head_q = np.zeros((TEST_END, 1))
    sc_para = list()
    for station in ['高道','飞来峡', '三水']:
        space = {'num_layers': hp.quniform('num_layers', 2, 3, 1),
                 'hidden_size': hp.quniform('hidden_size', 20, 100, 1),
                 'dropout': hp.quniform('dropout', 1, 46, 3),
                 }
        def objective(params):
            n = int(params['num_layers'])
            h = int(params['hidden_size'])
            d = int(params['dropout']) * 0.01
            l2 = 0
            if station == '三水':
                l2 = 0.01
            w = 12
            vnse = kfold_tlstm.kfold_vnse(n,h,d,l2,w,head_q, station,scenario_list=scenario, excel_file=EXCEL_FILE, batchsize=BATCHSIZE,
           start=START, train_end=TRAIN_END, k = K, fu_day=FU_DAY)
            gc.collect()
            return -vnse

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)
        print("Best Hyperparameters:", best)
        n = int(best['num_layers'])
        h = int(best['hidden_size'])
        d = best['dropout'] * 0.01
        l2 = 0
        if station == '三水':
            l2 = 0.01
        w = 12
        sc_para.append([n,h,d,l2,w])
        station_test, head_q = kfold_tlstm.station(n,h,d,l2,w,head_q,station,scenario_list=scenario, excel_file=EXCEL_FILE, batchsize=BATCHSIZE,
           start=START, train_end=TRAIN_END, test_end=TEST_END, fu_day=FU_DAY)
    print(sc_para)
    a_sim = np.empty([72, 0])
    b_sim = np.empty([72, 0])
    c_sim = np.empty([72, 0])
    for i in range(100):
        [n1,h1,d1,l21,w1] = sc_para[0]
        [n2,h2,d2,l22,w2] = sc_para[1]
        [n3,h3,d3,l23,w3] = sc_para[2]
        a_test_out, b_test_out, c_test_out = kfold_tlstm.kfold_tlstm(n1, h1, d1, l21, w1, n2, h2, d2, l22, w2, n3, h3, d3, l23, w3,excel_file=EXCEL_FILE, batchsize=BATCHSIZE,
           start=START, train_end=TRAIN_END, test_end=TEST_END, fu_day=FU_DAY,scenario_list=scenario)
        a_sim = np.concatenate((a_sim, a_test_out), axis=1)
        b_sim = np.concatenate((b_sim, b_test_out), axis=1)
        c_sim = np.concatenate((c_sim, c_test_out), axis=1)

    sc_para = pd.DataFrame(sc_para, columns=['num_layers', 'hidden_size', 'dropout', 'l2', 'windows'])
    names = ['高道','飞来峡','三水']
    sc_para.insert(0, '站名', names)
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(sc_para).to_excel(writer, sheet_name='参数', index=False)
        pd.DataFrame(a_sim).to_excel(writer, sheet_name='高道', index=False)
        pd.DataFrame(b_sim).to_excel(writer, sheet_name='飞来峡', index=False)
        pd.DataFrame(c_sim).to_excel(writer, sheet_name='三水', index=False)