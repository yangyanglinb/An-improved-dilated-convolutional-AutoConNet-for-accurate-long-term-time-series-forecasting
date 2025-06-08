import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class MovingAvg(nn.Module):
    """
    Moving average block 用来提取时间序列的长期趋势。
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pooled = self.avg(x_pad.permute(0, 2, 1))
        return x_pooled.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """
    Series decomposition: x = (x - moving_avg) + moving_avg
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class ResidualDilatedBlock(nn.Module):
    """
    单个 dilated‐conv block，带残差分支 + LayerNorm。
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super(ResidualDilatedBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        res = self.shortcut(x) if self.shortcut is not None else x
        out = self.conv(x) + res
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        return out.permute(0, 2, 1)


class ConvEncoder(nn.Module):
    """
    改进版的 Dilated‐conv Encoder：多层 ResidualDilatedBlock + 1×1 conv → repr_dims。
    """
    def __init__(self, in_channels, hidden_dims, repr_dims, depth, kernel_size=3):
        super(ConvEncoder, self).__init__()
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_dims
            layers.append(ResidualDilatedBlock(in_ch, hidden_dims, kernel_size, dilation))
        layers.append(nn.Conv1d(hidden_dims, repr_dims, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """
    AutoConNet_Improved：
      1) DataEmbedding → ConvEncoder
      2) Bottleneck MLP → length_mlp
      3) Multi‐scale trend MLP + SeriesDecomp
      4) Seasonality MLP + LayerNorm
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dims = configs.d_model
        self.repr_dims   = configs.d_ff
        self.depth       = configs.e_layers
        self.c_out       = configs.c_out

        self.AutoCon       = configs.AutoCon
        self.AutoCon_wnorm = configs.AutoCon_wnorm
        self.AutoCon_multiscales = configs.AutoCon_multiscales

        # 1) embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, self.hidden_dims,
            configs.embed, configs.freq, dropout=configs.dropout
        )

        # 2) conv encoder
        self.feature_extractor = ConvEncoder(
            in_channels=self.hidden_dims,
            hidden_dims=self.hidden_dims,
            repr_dims=self.repr_dims,
            depth=self.depth
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # 3) bottleneck MLP
        self.bottleneck = nn.Sequential(
            nn.Linear(self.repr_dims, self.repr_dims),
            nn.GELU(),
            nn.Linear(self.repr_dims, self.repr_dims),
        )

        # 4) length projection
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)

        # 5) multi‐scale trend
        self.trend_decoms = nn.ModuleList([
            SeriesDecomp(kernel_size=dlen + 1)
            for dlen in self.AutoCon_multiscales
        ])
        self.ch_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.repr_dims, self.repr_dims // 2),
                nn.GELU(),
                nn.Linear(self.repr_dims // 2, self.c_out)
            )
            for _ in self.AutoCon_multiscales
        ])

        # 6) input decomposition
        self.input_decom = SeriesDecomp(kernel_size=25)

        # 7) seasonality
        self.season_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.GELU(),
            nn.Linear(self.pred_len, self.pred_len)
        )
        self.season_norm = nn.LayerNorm(self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # 1) wnorm
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(1, keepdim=True).detach()
            seq_std  = x_enc.std(1, keepdim=True).detach()
            short_x  = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x   = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(1, keepdim=True).detach()
            short_x  = x_enc - seq_mean
            long_x   = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        else:
            seq_last = x_enc[:, -1:, :].detach()
            short_x  = x_enc - seq_last
            long_x   = short_x.clone()

        # 2) embed + conv
        enc_out = self.enc_embedding(long_x, x_mark_enc).transpose(1, 2)
        repr    = self.repr_dropout(self.feature_extractor(enc_out)).transpose(1, 2)

        if onlyrepr:
            return None, repr

        # 3) bottleneck
        repr = self.bottleneck(repr)

        # 4) length proj prep
        len_in = F.gelu(repr.transpose(1, 2))
        T_len = len_in.size(-1)
        if T_len != self.seq_len:
            diff = self.seq_len - T_len
            if diff > 0:
                pl = diff // 2
                pr = diff - pl
                len_in = F.pad(len_in, (pl, pr), mode='replicate')
            else:
                start = (-diff) // 2
                len_in = len_in[:, :, start:start + self.seq_len]
        len_out = self.length_mlp(len_in).transpose(1, 2)

        # 5) multi-scale trend
        trend_outs = []
        for decom, mlp in zip(self.trend_decoms, self.ch_mlps):
            _, d = decom(len_out)
            d = F.gelu(d)
            d = mlp(d)
            _, t = decom(d)
            trend_outs.append(t)
        trend_outs = torch.stack(trend_outs, -1).sum(-1)

        # 6) seasonality (pad/slice)
        season_in = short_x.permute(0, 2, 1)
        T_sea = season_in.size(-1)
        if T_sea != self.seq_len:
            diff = self.seq_len - T_sea
            if diff > 0:
                pl = diff // 2
                pr = diff - pl
                season_in = F.pad(season_in, (pl, pr), mode='replicate')
            else:
                start = (-diff) // 2
                season_in = season_in[:, :, start:start + self.seq_len]
        season_mid = self.season_mlp(season_in)    # (B, enc_in, pred_len)
        season_out = season_mid.permute(0, 2, 1)   # (B, pred_len, enc_in)
        season_out = self.season_norm(season_out)

        # 7) combine
        if self.AutoCon_wnorm == 'ReVIN':
            pred = (season_out + trend_outs) * (seq_std + 1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = season_out + trend_outs + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = season_out + trend_outs
        else:
            pred = season_out + trend_outs + seq_last

        return (pred, repr) if self.AutoCon else pred


series_decomp = SeriesDecomp
