import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True)
        # 定义全连接层，输出类别数为num_classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM输出维度是hidden_size*2

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 双向所以*2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))
        # 取序列的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out