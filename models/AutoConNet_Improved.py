

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
        # x: (B, T, C)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)  # (B, T + kernel_size - 1, C)
        x_pooled = self.avg(x_pad.permute(0, 2, 1))  # → (B, C, T)
        return x_pooled.permute(0, 2, 1)  # → (B, T, C)


class SeriesDecomp(nn.Module):
    """
    Series decomposition: x = (x - moving_avg) + moving_avg
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        # x: (B, T, C)
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
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        # 如果输入输出 channel 不同，需要先做 1×1 卷积投影
        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
        else:
            self.shortcut = None
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        # x: (B, in_ch, T)
        res = x
        out = self.conv(x)  # (B, out_ch, T)
        if self.shortcut is not None:
            res = self.shortcut(res)  # (B, out_ch, T)
        out = out + res             # (B, out_ch, T)
        # LayerNorm 需要在 (B, T, out_ch) 上做，所以先 permute 再归一化
        out = out.permute(0, 2, 1)  # (B, T, out_ch)
        out = self.norm(out)
        return out.permute(0, 2, 1)  # (B, out_ch, T)


class ConvEncoder(nn.Module):
    """
    改进版的 Dilated‐conv Encoder：每层都是 ResidualDilatedBlock，然后再用一层 1×1 卷积投到 repr_dims。
    """
    def __init__(self, in_channels, hidden_dims, repr_dims, depth, kernel_size=3):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_dims
            out_ch = hidden_dims
            layers.append(ResidualDilatedBlock(in_ch, out_ch, kernel_size, dilation))
        # 最后一层 1×1 卷积把 hidden_dims → repr_dims
        layers.append(nn.Conv1d(in_channels=hidden_dims, out_channels=repr_dims, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, hidden_dims, T)  → 输出 (B, repr_dims, T)
        return self.net(x)


class Model(nn.Module):
    """
    AutoConNet_Improved：
      1) DataEmbedding → ConvEncoder (残差+LayerNorm)
      2) Bottleneck MLP (两层) → 时间投影 length_mlp
      3) Multiscale trend‐MLP（两层→Linear）+ SeriesDecomp
      4) Seasonal 部分改为 double‐MLP + LayerNorm
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in   # 输入通道
        self.c_out = configs.c_out       # 输出通道
        self.hidden_dims = configs.d_model
        self.repr_dims = configs.d_ff
        self.depth = configs.e_layers

        self.AutoCon = configs.AutoCon
        self.AutoCon_wnorm = configs.AutoCon_wnorm
        self.AutoCon_multiscales = configs.AutoCon_multiscales

        # 1) Embedding: (B, T_in, enc_in) → (B, T_in, d_model)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, self.hidden_dims,
            configs.embed, configs.freq, dropout=configs.dropout
        )

        # 2) ConvEncoder (残差 + LayerNorm)
        self.feature_extractor = ConvEncoder(
            in_channels=self.hidden_dims,
            hidden_dims=self.hidden_dims,
            repr_dims=self.repr_dims,
            depth=self.depth,
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

        # 3) Bottleneck MLP: (B, T_in, repr_dims) → (B, T_in, repr_dims)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.repr_dims, self.repr_dims),
            nn.GELU(),
            nn.Linear(self.repr_dims, self.repr_dims),
        )

        # 4) Length‐projection: (B, repr_dims, T_in) → (B, repr_dims, pred_len)
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)

        # 5) Multiscale trend 重建：先做 SeriesDecomp 再 MLP 再分解提 trend
        self.trend_decoms = nn.ModuleList([
            SeriesDecomp(kernel_size=(dlen + 1))
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

        # 6) 输入分解（仅在 wnorm='Decomp' 时用）
        self.input_decom = SeriesDecomp(kernel_size=25)

        # 7) Seasonal 部分：双线性层 + LayerNorm
        self.season_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.GELU(),
            nn.Linear(self.pred_len, self.pred_len)
        )
        self.season_norm = nn.LayerNorm(self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        """
        x_enc:      (B, T_in, enc_in)
        x_mark_enc: (B, T_in, 时间特征维度)
        x_dec, x_mark_dec 在这个模型里不用，但为了和 Exp_Basic 保持一致必须写在签名里。
        """
        # 1) Window‐normalization → short_x, long_x
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()   # (B,1,enc_in)
            seq_std = x_enc.std(dim=1, keepdim=True).detach()     # (B,1,enc_in)
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:, -1:, :].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception(f"Unsupported wnorm: {self.AutoCon_wnorm}")

        # 2) Embedding + ConvEncoder → repr (B, T_in, repr_dims)
        enc_out = self.enc_embedding(long_x, x_mark_enc)  # (B, T_in, d_model)
        enc_out = enc_out.transpose(1, 2)                 # (B, d_model, T_in)
        repr = self.repr_dropout(self.feature_extractor(enc_out))  # (B, repr_dims, T_in)
        repr = repr.transpose(1, 2)                       # (B, T_in, repr_dims)

        if onlyrepr:
            return None, repr

        # 3) Bottleneck MLP: (B, T_in, repr_dims) → (B, T_in, repr_dims)
        repr = self.bottleneck(repr)

        # 4) Length‐projection: repr → (B, repr_dims, pred_len) → (B, pred_len, repr_dims)
        len_in = repr.transpose(1, 2)             # (B, repr_dims, T_in)
        len_mid = F.gelu(len_in)                  # (B, repr_dims, T_in)
        len_proj = self.length_mlp(len_mid)       # (B, repr_dims, pred_len)
        len_out = len_proj.transpose(1, 2)        # (B, pred_len, repr_dims)

        # 5) Multiscale trend 重建
        trend_outs = []
        for trend_decom, ch_mlp in zip(self.trend_decoms, self.ch_mlps):
            # len_out: (B, pred_len, repr_dims)
            _, dec_out = trend_decom(len_out)        # dec_out = 残差 → (B, pred_len, repr_dims)
            dec_out = F.gelu(dec_out)
            dec_out = ch_mlp(dec_out)                # (B, pred_len, c_out)
            _, trend_part = trend_decom(dec_out)     # 从 dec_out 中再分解出 trend
            trend_outs.append(trend_part)

        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)  # (B, pred_len, c_out)

        # 6) Seasonal 部分：采用双线性 → LayerNorm
        #    short_x: (B, T_in, enc_in) → (B, enc_in, T_in)
        season_in = short_x.permute(0, 2, 1)        # (B, enc_in, T_in)
        season_mid = self.season_mlp(season_in)    # (B, enc_in, pred_len)
        # → (B, pred_len, enc_in)
        season_mid = season_mid.permute(0, 2, 1)    # (B, pred_len, enc_in)
        # 假设 enc_in == c_out，否则这里需要加一个线性再映射到 c_out
        season_out = self.season_norm(season_mid)  # (B, pred_len, c_out)

        # 7) 最终合并 trend + season → pred
        if self.AutoCon_wnorm == 'ReVIN':
            pred = (season_out + trend_outs) * (seq_std + 1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = season_out + trend_outs + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = season_out + trend_outs
        elif self.AutoCon_wnorm == 'LastVal':
            pred = season_out + trend_outs + seq_last
        else:
            raise Exception(f"Unsupported wnorm: {self.AutoCon_wnorm}")

        if self.AutoCon:
            # 若做对比学习，需要同时返回 repr
            return pred, repr
        else:
            return pred

    def get_embeddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                       enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        """
        与 forward() 前半部分一致，但只返回 repr。
        """
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.AutoCon_wnorm == 'LastVal':
            seq_last = x_enc[:, -1:, :].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception(f"Unsupported wnorm: {self.AutoCon_wnorm}")

        enc_out = self.enc_embedding(long_x, x_mark_enc)   # (B, T_in, d_model)
        enc_out = enc_out.transpose(1, 2)                  # (B, d_model, T_in)
        repr = self.repr_dropout(self.feature_extractor(enc_out))  # (B, repr_dims, T_in)
        repr = repr.transpose(1, 2)                        # (B, T_in, repr_dims)
        repr = self.bottleneck(repr)                       # Bottleneck MLP 一致化
        return repr


# 兼容旧接口：如果其他地方 import series_decomp 依然有效
series_decomp = SeriesDecomp
