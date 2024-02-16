import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        # Weights initialization
        def _weights_init(m):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.GRU,
                              nn.LSTM)):
                print(m)
                nn.init.xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        x = self.dropout(self.relu(self.conv(x)))
        return x


class Encoder(LightningModule):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        #input shape: Batch x Channel x Time (Time has to be same size, channel = 6) 

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32,
                               kernel_size=24)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=16)
        self.conv3 = ConvBlock(in_channels=64, out_channels=96, kernel_size=8)
        self.out_size = 96

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global Max Pooling (as per
        # https://github.com/keras-team/keras/blob        # /7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/layers/pooling.py
        # #L559) for 'channels_first'
        x = F.max_pool1d(x, kernel_size=x.shape[2])
        #removed 2 inside squeeze to get shape 27
        x = x.squeeze()

        return x

