import torch
import torchvision
import torch.nn as nn
import numpy as np
import math


class Attn_2inp(nn.Module):
    def __init__(self, num_heads=2):
        super(Attn_2inp, self).__init__()

        dd = 4
        self.embed1 = nn.Linear(1, dd)
        self.embed2 = nn.Linear(1, dd)
        self.embed3 = nn.Linear(1, dd)
        self.mulhead_attn1 = nn.MultiheadAttention(embed_dim=dd, num_heads=num_heads, batch_first=True)
        self.activate1 = nn.Tanh()
        self.mulhead_attn2 = nn.MultiheadAttention(embed_dim=dd, num_heads=num_heads, batch_first=True)
        self.activate2 = nn.Tanh()
        self.output = nn.Sequential(
            nn.Linear(dd, 16),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(16, 4),
            # nn.ReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(4, 1)
        )
        # self._initial_parameters()
        

    def forward(self, x):
        x0 = self.embed1(x[0])
        x1 = self.embed2(x[1].reshape((x[1].shape[0], x[1].shape[1] * x[1].shape[2], 1)))
        x2 = self.embed3(x[0])
        x, _ = self.mulhead_attn1(x0, x2, x1)
        # x = self.activate1(x)
        # x, _ = self.mulhead_attn2(x, x, x)
        # x = self.activate2(x)
        x = self.output(x)
        return x
    

class Basic_3inp(nn.Module):
    def __init__(self, kernel_size=3, drop_ratio=0.1) -> None:
        super(Basic_3inp, self).__init__()
        self.cnnlayer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            # nn.Dropout(drop_ratio),
            nn.Conv1d(32, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            # nn.Dropout(drop_ratio),
            nn.Conv1d(32, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(5),
            # nn.Dropout(drop_ratio)
            nn.Conv1d(32, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh()
        )

        self.cnnlayer2 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_ratio),
            nn.Conv1d(4, 8, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_ratio),
            nn.Conv1d(8, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(5),
            nn.Dropout(drop_ratio)
        )

        self.mlplayer = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 40),
            nn.Tanh()
        )

        self.fclayer = nn.Sequential(
            nn.Linear(280, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 200)
        )

    
    def forward(self, x):
        y1 = self.cnnlayer1(x[0].permute(0, 2, 1))
        y2 = self.cnnlayer2(x[1].permute(0, 2, 1))
        y3 = self.mlplayer(x[2].permute(0, 2, 1))
        y1 = y1.reshape((y1.shape[0], -1))
        y2 = y2.reshape((y2.shape[0], -1))
        y3 = y3.reshape((y3.shape[0], -1))
        y = torch.concat([y1, y2, y3], dim=1)
        y = self.fclayer(y)
        return y[:, :, None]
    

class Basic_2inp(nn.Module):
    def __init__(self, kernel_size=5, drop_ratio=0.2) -> None:
        super(Basic_2inp, self).__init__()
        self.cnnlayer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            # nn.Dropout(drop_ratio),
            nn.Conv1d(32, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            # nn.Dropout(drop_ratio),
            nn.Conv1d(32, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(5),
            # nn.Dropout(drop_ratio)
            nn.Conv1d(32, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh()
        )

        self.cnnlayer2 = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_ratio),
            nn.Conv1d(4, 8, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_ratio),
            nn.Conv1d(8, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(5),
            nn.Dropout(drop_ratio)
        )

        self.fclayer = nn.Sequential(
            nn.Linear(240, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 200)
        )


    def forward(self, x):
        y1 = self.cnnlayer1(x[0].permute(0, 2, 1))
        y2 = self.cnnlayer2(x[1].permute(0, 2, 1))
        y1 = y1.reshape((y1.shape[0], -1))
        y2 = y2.reshape((y2.shape[0], -1))
        y = torch.concat([y1, y2], dim=1)
        y = self.fclayer(y)
        return y[:, :, None]


class Basic_1inp(nn.Module):
    def __init__(self, kernel_size=5, drop_ratio=0.2) -> None:
        super(Basic_1inp, self).__init__()
        self.cnnlayer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            # nn.Dropout(drop_ratio),
            nn.Conv1d(32, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            # nn.Dropout(drop_ratio),
            nn.Conv1d(32, 32, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(5),
            # nn.Dropout(drop_ratio)
            nn.Conv1d(32, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh()
        )


        self.fclayer = nn.Sequential(
            nn.Linear(160, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 200)
        )


    def forward(self, x):
        y = self.cnnlayer1(x[0].permute(0, 2, 1))
        y = y.reshape((y.shape[0], -1))
        y = self.fclayer(y)
        return y[:, :, None]


class CNN_basic(nn.Module):
    def __init__(self, input_size=2, kernel_size=3, drop_ratio=0.1):
        super(CNN_basic, self).__init__()
        self.kernel_size = kernel_size
        self.cnnlayer = nn.Sequential(
            nn.Conv1d(input_size, 4, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_ratio),
            nn.Conv1d(4, 8, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(2),
            nn.Dropout(drop_ratio),
            nn.Conv1d(8, 16, kernel_size, bias=True, padding=int((kernel_size - 1) / 2)),
            nn.Tanh(),
            nn.MaxPool1d(5),
            nn.Dropout(drop_ratio)
        )

        self.fc = nn.Sequential(
            nn.Linear(80, 128),
            nn.Dropout(drop_ratio),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Dropout(drop_ratio),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Dropout(drop_ratio),
            nn.Tanh(),
            nn.Linear(128, 100)
        )


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnnlayer(x)
        x = x.permute(0, 2, 1)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x[:, :, None]
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.01)
            else:
                nn.init.constant_(p, 0)


class MLP(nn.Module):
    def __init__(self, lens=202):
        super(MLP, self).__init__()
        
        self.lens = lens
        self.fc = nn.Sequential(
            nn.Linear(self.lens, 256),
            # nn.Dropout(0.15),
            nn.Tanh(),
            nn.Linear(256, 128),
            # nn.Dropout(0.15),
            nn.Tanh(),
            nn.Linear(128, 64),
            # nn.Dropout(0.15),
            nn.Tanh(),
            nn.Linear(64, 100),
            nn.Tanh()
        )
        # self._initial_parameters()
        

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))[:, :, None]
        x = x.permute(0, 2, 1)
        x = self.fc(x)        
        return x.permute(0, 2, 1)
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.05)
            else:
                nn.init.constant_(p, 0)


class MLP_basic(nn.Module):
    def __init__(self, lens=101):
        super(MLP_basic, self).__init__()
        
        self.lens = lens
        self.fc = nn.Sequential(
            nn.Linear(self.lens, 256),
            nn.Dropout(0.15),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Dropout(0.15),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(0.15),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Dropout(0.15),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.Dropout(0.15),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.ReLU(inplace=True)
        )
        # self._initial_parameters()
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)        
        return x.permute(0, 2, 1)
    
    
    def _initial_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.constant_(p, 0.05)
            else:
                nn.init.constant_(p, 0)