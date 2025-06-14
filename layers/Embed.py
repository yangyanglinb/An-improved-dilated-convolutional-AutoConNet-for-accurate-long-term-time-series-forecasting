import torch
import torch.nn as nn
import numpy as np
import math

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TemporalEmbedding, self).__init__()
        freq_map = {
            'h': ['hour', 'day', 'weekday', 'month'],
            't': ['minute', 'hour', 'day', 'weekday'],
            'd': ['day', 'weekday', 'month'],
            'm': ['month']
        }

        freq_dim = freq_map.get(freq, ['hour', 'day', 'weekday', 'month'])
        self.embed_type = embed_type
        self.d_model = d_model
        self.embed_layers = nn.ModuleDict()

        for name in freq_dim:
            self.embed_layers[name] = nn.Embedding(32, d_model)  # Max size = 31

    def forward(self, x):
        # x: [batch, seq_len, num_time_features], e.g. [32, 96, 4]
        emb = 0
        for i, (name, emb_layer) in enumerate(self.embed_layers.items()):
            emb += emb_layer(x[:, :, i].long())  # Apply embedding for each feature
        return emb

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) +             self.position_embedding(x) +             self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, c_in, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Conv1d(in_channels=c_in,
                                    out_channels=d_model,
                                    kernel_size=patch_len,
                                    stride=stride)

    def forward(self, x):
        # x: [Batch, SeqLen, Channels] -> [Batch, Channels, SeqLen]
        x = x.permute(0, 2, 1)
        # Apply Conv1D projection
        x = self.projection(x)  # -> [Batch, d_model, NumPatches]
        # Rearrange back to [Batch, NumPatches, d_model]
        x = x.transpose(1, 2)
        return x
