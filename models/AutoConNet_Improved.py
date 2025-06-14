import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Embed import DataEmbedding


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rates=[1, 2, 4], dropout=0.1):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=d, dilation=d)
            for d in dilation_rates
        ])
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, C, L]
        out = sum(conv(x) for conv in self.convs) / len(self.convs)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(MultiScaleConvBlock, self).__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: [B, C, L]
        out = sum(branch(x) for branch in self.branches) / len(self.branches)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, D, L] -> [B, L, D]
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x.permute(0, 2, 1)  # Back to [B, D, L]
        return x

class AutoConNet_Improved(nn.Module):
    def __init__(self, configs):
        super(AutoConNet_Improved, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_attention = getattr(configs, 'use_attention', True)

        # Input embedding
        from layers.Embed import DataEmbedding
        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )

        # Dilated Convolution block
        self.dilated_conv = DilatedConvBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            dilation_rates=[1, 2, 4],
            dropout=configs.dropout
        )

        # Multi-scale Convolution block
        self.multiscale_conv = MultiScaleConvBlock(
            in_channels=configs.d_model,
            out_channels=configs.d_model,
            kernel_sizes=[3, 5, 7],
            dropout=configs.dropout
        )

        # Optional attention
        self.attn = SelfAttention(d_model=configs.d_model, n_heads=configs.n_heads) if self.use_attention else nn.Identity()

        # Output projection
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [B, L, C]
        enc_out = self.embedding(x_enc, x_mark_enc)  # [B, L, D]
        enc_out = enc_out.permute(0, 2, 1)  # [B, D, L]

        conv_out1 = self.dilated_conv(enc_out)
        conv_out2 = self.multiscale_conv(enc_out)
        combined = (conv_out1 + conv_out2) / 2  # [B, D, L]

        combined = self.attn(combined)  # Optional attention
        combined = combined.permute(0, 2, 1)  # [B, L, D]

        output = self.projection(combined[:, -self.pred_len:, :])  # [B, pred_len, C]
        return output

Model = AutoConNet_Improved
