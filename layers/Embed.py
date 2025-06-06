import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False
        )
        nn.init.kaiming_normal_(self.tokenConv.weight,
                                mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    old style “one-hot + embed” 时间特征
    """
    def __init__(self, d_model: int, embed_type='fixed', freq='h', wo_freq=' '):
        super().__init__()

        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        if 'h' not in wo_freq:
            self.hour_embed = Embed(hour_size, d_model)
        if 'd' not in wo_freq:
            self.day_embed = Embed(day_size, d_model)
        if 'w' not in wo_freq:
            self.weekday_embed = Embed(weekday_size, d_model)
        if 'm' not in wo_freq:
            self.month_embed = Embed(month_size, d_model)

        print(f'TemporalEmbedding({embed_type})-wo_freq: {wo_freq}')

    def forward(self, x):
        x = x.long()
        minute_x  = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x    = self.hour_embed(x[:, :, 3])   if hasattr(self, 'hour_embed') else 0.
        weekday_x = self.weekday_embed(x[:, :, 2]) if hasattr(self, 'weekday_embed') else 0.
        day_x     = self.day_embed(x[:, :, 1])    if hasattr(self, 'day_embed') else 0.
        month_x   = self.month_embed(x[:, :, 0])  if hasattr(self, 'month_embed') else 0.
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    用连续时间特征做线性映射：
    t>h>w>d>m>q>y 依次拼接；wo_freq 可以屏蔽任意维
    """
    def __init__(self, d_model: int, embed_type='timeF', freq='h', wo_freq=' '):
        super().__init__()

        # 维度顺序与 utils/timefeatures.py 保持一致
        freq_dim_map = {'h': 0, 'w': 1, 'd': 2, 'm': 3, 'q': 4, 'y': 5}
        self.rm_idx = [freq_dim_map[f] for f in (wo_freq or '') if f in freq_dim_map]

        # 每种粒度输入维度数量
        freq_map = {
            'h': 4,   # hour
            't': 5,   # minute-level
            's': 6,   # second-level
            'm': 1, 'a': 1,             # month
            'q': 1,                      # quarter  (NEW)
            'y': 1,                      # year     (NEW)
            'w': 2,                      # week
            'd': 3, 'b': 3               # day / business-day
        }
        self.d_inp = freq_map[freq]
        self.embed = nn.Linear(self.d_inp, d_model, bias=False)

        print('TimeFeatureEmbedding-wo-freq:', wo_freq, 'rm_idx=', self.rm_idx)

    def forward(self, x):
        if self.rm_idx:
            x[:, :, self.rm_idx] = 0.0
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed',
                 freq='h', wo_freq=' ', dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model, embed_type, freq, wo_freq)
                                   if embed_type != 'timeF'
                                   else TimeFeatureEmbedding(d_model, embed_type, freq, wo_freq))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (self.value_embedding(x)
                 + self.temporal_embedding(x_mark)
                 + self.position_embedding(x))
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model, embed_type, freq, ' ')
                                   if embed_type != 'timeF'
                                   else TimeFeatureEmbedding(d_model, embed_type, freq, ' '))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    TimesNet patch-level embedding
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
