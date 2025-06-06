# models/AutoConNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of a time series.
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, T, C)
        # pad “front” and “end” so that the pooled result has the same length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)       # (B, T + kernel_size - 1, C)
        x_pooled = self.avg(x_pad.permute(0, 2, 1))     # → (B, C, T)
        return x_pooled.permute(0, 2, 1)                # → (B, T, C)


class SeriesDecomp(nn.Module):
    """
    Simple decomposition into “residual + moving average”:
      x = (x - moving_avg) + moving_avg.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        # x: (B, T, C)
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block:
      - 对 repr 做全局平均池化 → 两层小型 MLP → 通道注意力权重
    """
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # time → 1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: (B, T, C)
        b, t, c = x.size()
        # 先 permute → (B, C, T)，avg over T → (B, C, 1)
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)  # (B, C)
        y = self.fc(y)                                    # (B, C), 在 [0,1]
        y = y.view(b, 1, c)                               # (B,1,C)
        out = x * y                                       # 广播 (B, T, C)
        return self.dropout(out)


class ConvEncoder(nn.Module):
    """
    A simple dilated‐conv “encoder” that stacks a few 1D convolutional layers with exponentially
    increasing dilation rates, then projects to repr_dims at the end.

    - in_channels: 输入通道数 (→ hidden_dims)
    - hidden_dims: 中间卷积层宽度
    - repr_dims: 最终“瓶颈”通道数
    - depth: 堆叠多少层 dilated conv
    - kernel_size: 卷积核大小（默认 3）
    """
    def __init__(self, in_channels, hidden_dims, repr_dims, depth, kernel_size=3):
        super(ConvEncoder, self).__init__()
        self.depth = depth
        self.hidden_dims = hidden_dims
        self.repr_dims = repr_dims
        self.kernel_size = kernel_size

        layers = []
        for i in range(depth):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            in_ch = in_channels if i == 0 else hidden_dims
            out_ch = hidden_dims
            layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ))
            layers.append(nn.ReLU(inplace=True))

        # 最终把 hidden_dims 投影到 repr_dims（1×1 卷积）
        layers.append(nn.Conv1d(
            in_channels=hidden_dims,
            out_channels=repr_dims,
            kernel_size=1
        ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, hidden_dims, T)   ← 来自 AutoConNet 的调用
        return: (B, repr_dims, T)
        """
        return self.net(x)


class Model(nn.Module):
    """
    AutoConNet 再次改进版，新增 SEBlock + 更深的 TransformerEncoder。

    参数说明：
        configs.seq_len:     输入序列长度 (T_in)
        configs.label_len:   “解码器预热”长度，仅用于构造 dec_inp
        configs.pred_len:    预测长度
        configs.enc_in:      输入通道数（特征数）
        configs.c_out:       输出通道数（通常为 1）
        configs.d_model:     embedding 和卷积堆栈的 hidden_dims
        configs.d_ff:        repr_dims（卷积堆栈最后的瓶颈通道数）
        configs.e_layers:    ConvEncoder 堆叠层数
        configs.AutoCon:     是否开启对比学习分支（返回 (pred, repr)）
        configs.AutoCon_wnorm: one of {"ReVIN","Mean","Decomp","LastVal"}
        configs.AutoCon_multiscales: list，例如 [96, 192, 336]
        configs.dropout:     embedding 处的 dropout
        configs.attn_heads:  Transformer Encoder 中的头数
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.c_out = configs.c_out
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
        self.embedding_norm = nn.LayerNorm(self.hidden_dims)

        # 2) DilatedConvEncoder → repr_conv
        self.feature_extractor = ConvEncoder(
            in_channels=self.hidden_dims,
            hidden_dims=self.hidden_dims,
            repr_dims=self.repr_dims,
            depth=self.depth,
            kernel_size=3
        )
        self.post_conv_norm = nn.LayerNorm(self.repr_dims)
        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)

        # 3) TransformerEncoder（深化到 2 层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.repr_dims,
            nhead=getattr(configs, 'attn_heads', 4),
            dim_feedforward=self.repr_dims * 2,
            dropout=0.1,
            activation='gelu'
        )
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.trans_norm = nn.LayerNorm(self.repr_dims)

        # 4) SE 注意力：对 repr 做通道注意力
        self.se_block = SEBlock(channel=self.repr_dims, reduction=8)

        # 5) 对每个尺度，用两层 MLP 预测 c_out
        self.ch_mlps = nn.ModuleList()
        for _ in self.AutoCon_multiscales:
            self.ch_mlps.append(
                nn.Sequential(
                    nn.Linear(self.repr_dims, self.repr_dims),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.1),
                    nn.Linear(self.repr_dims, self.c_out)
                )
            )

        # 6) Length‐projection MLP: (seq_len → pred_len)，应用在“时间”维度
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)

        # 7) 每个尺度做 Series Decomposition
        self.trend_decoms = nn.ModuleList([
            SeriesDecomp(kernel_size=(dlen + 1))
            for dlen in self.AutoCon_multiscales
        ])

        # 8) 可选：先对输入做一次分解（仅在 Decomp 模式使用）
        self.input_decom = SeriesDecomp(kernel_size=25)

        # 9) 季节性分支：两层 MLP，从 short_x → pred_len
        self.season_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.pred_len, self.pred_len)
        )

        # 10) 最终对输出通道做 LayerNorm
        self.final_norm = nn.LayerNorm(self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        """
        x_enc:      (B, T_in, enc_in)
        x_mark_enc: (B, T_in, #time_features)
        x_dec, x_mark_dec: 保持接口兼容，不在本模型中使用
        """
        # 1) Window‐normalization（wnorm）
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()    # (B,1,enc_in)
            seq_std = x_enc.std(dim=1, keepdim=True).detach()      # (B,1,enc_in)
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
            raise Exception(
                f"Not Supported Window Normalization: {self.AutoCon_wnorm}. "
                f"Use one of {{'ReVIN','Mean','Decomp','LastVal'}}."
            )

        # 2) Embedding + ConvEncoder → repr_conv
        enc_out = self.enc_embedding(long_x, x_mark_enc)       # (B, T_in, d_model)
        enc_out = self.embedding_norm(enc_out)
        enc_out = enc_out.transpose(1, 2)                      # (B, d_model, T_in)
        repr_conv = self.feature_extractor(enc_out)            # (B, repr_dims, T_in)
        repr_conv = repr_conv.transpose(1, 2)                  # (B, T_in, repr_dims)
        repr_conv = self.post_conv_norm(repr_conv)
        repr_conv = self.repr_dropout(repr_conv)
        repr = self.repr_head(repr_conv)                       # (B, T_in, repr_dims)

        # 3) TransformerEncoderLayer × 2 + 残差 + LayerNorm
        #    Transformer 需要 (T_in, B, repr_dims) 形式
        repr_t = repr.transpose(0, 1)                          # (T_in, B, repr_dims)
        repr_t = self.trans_encoder(repr_t)                    # (T_in, B, repr_dims)
        repr_t = repr_t.transpose(0, 1)                         # (B, T_in, repr_dims)
        repr = self.trans_norm(repr + repr_t)                  # (B, T_in, repr_dims)

        # 4) SE 注意力
        repr = self.se_block(repr)                             # (B, T_in, repr_dims)

        if onlyrepr:
            return None, repr

        # 5) Length‐projection：repr (B, T_in, repr_dims) → (B, pred_len, repr_dims)
        len_in = repr.transpose(1, 2)                          # (B, repr_dims, T_in)
        len_out = F.gelu(len_in)                               # (B, repr_dims, T_in)
        len_out = self.length_mlp(len_out)                     # (B, repr_dims, pred_len)
        len_out = len_out.transpose(1, 2)                       # (B, pred_len, repr_dims)

        # 6) 多尺度趋势重建
        trend_outs = []
        for trend_decom, ch_mlp in zip(self.trend_decoms, self.ch_mlps):
            _, dec_out = trend_decom(len_out)                   # 去低频
            dec_out = F.gelu(dec_out)
            dec_out = ch_mlp(dec_out)                           # (B, pred_len, c_out)
            _, trend_only = trend_decom(dec_out)                # 提取趋势部分
            trend_outs.append(trend_only)
        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)  # (B, pred_len, c_out)

        # 7) 季节性分支
        season_in = short_x.permute(0, 2, 1)                    # (B, enc_in, T_in)
        season_part = self.season_mlp(season_in)                # (B, enc_in, pred_len)
        season_out = season_part.permute(0, 2, 1)                # (B, pred_len, enc_in)

        # 8) 趋势 + 季节 → 最终预测
        if self.AutoCon_wnorm == 'ReVIN':
            pred = (season_out + trend_outs) * (seq_std + 1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = season_out + trend_outs + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = season_out + trend_outs
        elif self.AutoCon_wnorm == 'LastVal':
            pred = season_out + trend_outs + seq_last
        else:
            raise Exception(f"Not Supported Window Normalization: {self.AutoCon_wnorm}")

        # 9) 最后做一次 LayerNorm
        B, L, C = pred.size()
        pred_flat = pred.reshape(-1, C)                         # (B*pred_len, c_out)
        pred_normed = self.final_norm(pred_flat)                # (B*pred_len, c_out)
        pred = pred_normed.reshape(B, L, C)                      # (B, pred_len, c_out)

        if self.AutoCon:
            return pred, repr
        else:
            return pred

    def get_embeddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                       enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        """
        与 forward 前半部分一致，只返回 repr（(B, T_in, repr_dims)）。
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
            raise Exception(f"Not Supported Window Normalization: {self.AutoCon_wnorm}")

        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = self.embedding_norm(enc_out)
        enc_out = enc_out.transpose(1, 2)
        repr_conv = self.feature_extractor(enc_out)
        repr_conv = repr_conv.transpose(1, 2)
        repr_conv = self.post_conv_norm(repr_conv)
        repr_conv = self.repr_dropout(repr_conv)
        repr = self.repr_head(repr_conv)

        repr_t = repr.transpose(0, 1)
        repr_t = self.trans_encoder(repr_t)
        repr_t = repr_t.transpose(0, 1)
        repr = self.trans_norm(repr + repr_t)

        # SEBlock
        repr = self.se_block(repr)

        return repr


# 向后兼容：保持旧版接口
series_decomp = SeriesDecomp
