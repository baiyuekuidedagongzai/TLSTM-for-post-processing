import torch
import torch.nn as nn

#==================================================一、创建多变量输入LSTM模型=================================================
class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, x):
        q = self.query(x).transpose(0, 1)       # 32,1452,116
        k = self.key(x).transpose(0, 1)       # 形状为 (batch_size, seq_len, embed_dim) 的张量 k：
        v = self.value(x).transpose(0, 1)
        att_scores = torch.matmul(q, k.transpose(1, 2))     # 32, 1452, 1452
        att_weights = nn.functional.softmax(att_scores, dim = -1)       # 32, 1452,1452
        att_outputs = torch.matmul(att_weights, v).transpose(0, 1)
        return att_outputs


class AttentionLSTM(nn.Module):
    # 不将batch_size硬编码为模型的属性的好处是，可以更灵活地处理不同批次大小的输入数据。这需要在 forward 方法中直接使用输入数据的批次大小，而不是依赖于模型的属性。
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)

        # 在PyTorch中，nn.LSTM 模块默认的 batch_first 参数是 False。
        # 当 batch_first 是 False 时，输入和输出张量的形状是 (sequence_length, batch_size, input_size)，其中 sequence_length 是时间步的数量，batch_size 是批次大小。
        # 如果不将 batch_first 设置为 True，则 nn.LSTM 会默认将第一个维度解释为时间步而不是批次，导致错误的形状匹配。
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        # LSTM层本身具有内部激活函数。在PyTorch中，默认情况下，LSTM层使用双曲正切（tanh）激活函数来更新隐藏状态，并使用Sigmoid激活函数来处理输入门、遗忘门和输出门。
        # 应用的ReLU激活是在LSTM层之后额外添加的激活。
        self.activation = nn.ReLU()
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.attention = SelfAttention(hidden_size)
    # forward方法是在调用模型进行前向传播时自动调用的。
    # 在训练过程中，可以通过传递输入数据到模型，模型会自动执行前向传播，并返回预测结果。在本代码中，训练过程中的前向传播部分在下面的循环中：outputs = model(inputs)
    def forward(self, input_seq):
        seq_len, batch_size = input_seq.shape[0], input_seq.shape[1]
        # 隐藏状态的预期形状为（层数 * 方向数，批次大小，隐藏单元大小）
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        output, _ = self.lstm(input_seq, (h0, c0))
        output = self.dropout(output)
        output = self.activation(output)
        pred = self.linear(output)
        pred = pred.view(seq_len, batch_size, -1)
        return pred
