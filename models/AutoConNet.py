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
        y = self.avg_pool(x.permute(0, 2, 1)).view(b, c)
        y = self.fc(y)
        y = y.view(b, 1, c)
        out = x * y
        return self.dropout(out)


class ConvEncoder(nn.Module):
    """
    A simple dilated‐conv “encoder” that stacks a few 1D convolutional layers with exponentially
    increasing dilation rates, then projects to repr_dims at the end.
    """
    def __init__(self, in_channels, hidden_dims, repr_dims, depth, kernel_size=3):
        super(ConvEncoder, self).__init__()
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            in_ch = in_channels if i == 0 else hidden_dims
            layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=hidden_dims,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(in_channels=hidden_dims, out_channels=repr_dims, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, hidden_dims, T) → (B, repr_dims, T)
        return self.net(x)


class Model(nn.Module):
    """
    AutoConNet 改进版，新增 SEBlock + 更深的 TransformerEncoder。
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

        # 1) Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, self.hidden_dims,
            configs.embed, configs.freq, dropout=configs.dropout
        )
        self.embedding_norm = nn.LayerNorm(self.hidden_dims)

        # 2) ConvEncoder
        self.feature_extractor = ConvEncoder(
            in_channels=self.hidden_dims,
            hidden_dims=self.hidden_dims,
            repr_dims=self.repr_dims,
            depth=self.depth
        )
        self.post_conv_norm = nn.LayerNorm(self.repr_dims)
        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)

        # 3) TransformerEncoder ×2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.repr_dims,
            nhead=getattr(configs, 'attn_heads', 4),
            dim_feedforward=self.repr_dims * 2,
            dropout=0.1,
            activation='gelu'
        )
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.trans_norm = nn.LayerNorm(self.repr_dims)

        # 4) SEBlock
        self.se_block = SEBlock(channel=self.repr_dims, reduction=8)

        # 5) per-scale channel MLPs
        self.ch_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.repr_dims, self.repr_dims),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(self.repr_dims, self.c_out)
            )
            for _ in self.AutoCon_multiscales
        ])

        # 6) time‐projection MLP
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)

        # 7) multi‐scale series decompositions
        self.trend_decoms = nn.ModuleList([
            SeriesDecomp(kernel_size=dlen + 1)
            for dlen in self.AutoCon_multiscales
        ])

        # 8) optional input decomposition
        self.input_decom = SeriesDecomp(kernel_size=25)

        # 9) seasonality branch
        self.season_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.pred_len, self.pred_len)
        )

        # 10) final output norm
        self.final_norm = nn.LayerNorm(self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # 1) window‐normalization
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std  = x_enc.std(dim=1, keepdim=True).detach()
            short_x  = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x   = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x  = (x_enc - seq_mean)
            long_x   = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        else:  # LastVal
            seq_last = x_enc[:, -1:, :].detach()
            short_x  = (x_enc - seq_last)
            long_x   = short_x.clone()

        # 2) embedding + conv encoder
        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = self.embedding_norm(enc_out).transpose(1, 2)
        repr_conv = self.feature_extractor(enc_out).transpose(1, 2)
        repr_conv = self.post_conv_norm(repr_conv)
        repr_conv = self.repr_dropout(repr_conv)
        repr = self.repr_head(repr_conv)

        # 3) transformer ×2 + residual
        repr_t = repr.transpose(0, 1)
        repr_t = self.trans_encoder(repr_t).transpose(0, 1)
        repr = self.trans_norm(repr + repr_t)

        # 4) SEBlock
        repr = self.se_block(repr)

        if onlyrepr:
            return None, repr

        # 5) length projection prep
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

        # 6) multi-scale trend
        trend_outs = []
        for decom, mlp in zip(self.trend_decoms, self.ch_mlps):
            _, d = decom(len_out)
            d = F.gelu(d)
            d = mlp(d)
            _, t = decom(d)
            trend_outs.append(t)
        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)

        # 7) seasonality branch (pad/slice to seq_len first)
        season_in = short_x.permute(0, 2, 1)  # (B, enc_in, T_enc)
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
        season_mid = self.season_mlp(season_in)     # (B, enc_in, pred_len)
        season_out = season_mid.permute(0, 2, 1)    # (B, pred_len, enc_in)

        # 8) combine trend + season
        if self.AutoCon_wnorm == 'ReVIN':
            pred = (season_out + trend_outs) * (seq_std + 1e-5) + seq_mean
        elif self.AutoCon_wnorm == 'Mean':
            pred = season_out + trend_outs + seq_mean
        elif self.AutoCon_wnorm == 'Decomp':
            pred = season_out + trend_outs
        else:
            pred = season_out + trend_outs + seq_last

        # 9) final norm
        B, L, C = pred.size()
        pred = self.final_norm(pred.reshape(-1, C)).reshape(B, L, C)

        return (pred, repr) if self.AutoCon else pred

    def get_embeddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                       enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):
        # same as forward up to repr
        if self.AutoCon_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std  = x_enc.std(dim=1, keepdim=True).detach()
            short_x  = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x   = short_x.clone()
        elif self.AutoCon_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x  = (x_enc - seq_mean)
            long_x   = short_x.clone()
        elif self.AutoCon_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        else:
            seq_last = x_enc[:, -1:, :].detach()
            short_x  = (x_enc - seq_last)
            long_x   = short_x.clone()

        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = self.embedding_norm(enc_out).transpose(1, 2)
        repr_conv = self.feature_extractor(enc_out).transpose(1, 2)
        repr_conv = self.post_conv_norm(repr_conv)
        repr_conv = self.repr_dropout(repr_conv)
        repr = self.repr_head(repr_conv)

        repr_t = repr.transpose(0, 1)
        repr_t = self.trans_encoder(repr_t).transpose(0, 1)
        repr = self.trans_norm(repr + repr_t)
        repr = self.se_block(repr)
        return repr


series_decomp = SeriesDecomp
