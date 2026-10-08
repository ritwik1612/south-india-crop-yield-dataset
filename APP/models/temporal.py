"""Paper-inspired sequence architectures, pending weather tensor preparation/training."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class YieldLSTM(nn.Module):
    """Padded weather [batch,time,10], true lengths, static/context features."""
    def __init__(self, static_dim, weather_dim=10, hidden_dim=64):
        super().__init__()
        self.lstm=nn.LSTM(weather_dim,hidden_dim,num_layers=2,batch_first=True,dropout=0.2)
        self.head=nn.Sequential(nn.Linear(hidden_dim+static_dim,64),nn.GELU(),
                                nn.Dropout(0.2),nn.Linear(64,1))
    def forward(self, weather, lengths, static):
        packed=pack_padded_sequence(weather,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.lstm(packed)
        return self.head(torch.cat([hidden[-1],static],dim=1)).squeeze(-1)

class YieldCNN(nn.Module):
    """Fixed-length weather [batch,time,10] plus static/context features."""
    def __init__(self, static_dim, weather_dim=10):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(weather_dim,32,3,padding=1),nn.GELU(),
                                    nn.Conv1d(32,64,3,padding=1),nn.GELU(),
                                    nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Linear(64+static_dim,64),nn.GELU(),
                                nn.Dropout(0.2),nn.Linear(64,1))
    def forward(self, weather, static):
        encoded=self.features(weather.transpose(1,2)).squeeze(-1)
        return self.head(torch.cat([encoded,static],dim=1)).squeeze(-1)
