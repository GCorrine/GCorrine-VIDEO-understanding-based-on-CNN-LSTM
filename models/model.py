import torch
import torch.nn as nn
import torchvision.models as models
from .lstm import LSTM

class CNNLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super(CNNLSTM, self).__init__()
        # 使用ResNet50直到avgpool层之前
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后两层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化
        self.lstm = LSTM(input_size=2048, hidden_size=hidden_size,
                        num_layers=num_layers, num_classes=num_classes)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)
        # 通过CNN提取特征
        cnn_out = self.cnn(x)  # [batch*seq, 2048, h', w']
        cnn_out = self.avgpool(cnn_out)  # [batch*seq, 2048, 1, 1]
        cnn_out = cnn_out.view(batch_size, seq_length, -1)  # [batch, seq, 2048]
        # 通过LSTM处理特征序列
        lstm_out = self.lstm(cnn_out)
        return lstm_out