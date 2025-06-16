import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class RobustDilatedConv(nn.Module):
    def __init__(self, d_model, dilation):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, 3, 
                            padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv(x).permute(0, 2, 1)  # [B, L, D]
        return self.dropout(F.gelu(self.norm(x))).permute(0, 2, 1)

class AutoConNet_Improved(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        # 输入嵌入（修正了括号问题）
        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model,
            configs.embed, configs.freq,
            max(0.1, configs.dropout)  # 防止过拟合
        )
        
        # 改进的卷积模块
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(configs.d_model, configs.d_model, 3, 
                         padding=d, dilation=d),
                nn.BatchNorm1d(configs.d_model),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for d in [1, 2, 4]  # 多尺度空洞卷积
        ])
        
        # 稳健输出层
        self.projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model//2),
            nn.GELU(),
            nn.Linear(configs.d_model//2, configs.c_out),
            nn.Tanhshrink()  # 输出范围: ~(-1,1)
        )
        
        # 自适应缩放
        self.scale = nn.Parameter(torch.tensor(0.5))  # 可学习参数

    def forward(self, x_enc, x_mark_enc, dec_inp=None, dec_mark=None):
        """兼容4参数调用"""
        # 嵌入层
        x = self.embedding(x_enc, x_mark_enc).permute(0, 2, 1)  # [B, D, L]
        
        # 多尺度特征融合
        conv_out = torch.stack([block(x) for block in self.conv_blocks]).mean(dim=0)
        
        # 稳健输出
        output = self.projection(conv_out.permute(0, 2, 1)[:, -self.configs.pred_len:, :])
        return output * self.scale

# 保持框架兼容性
Model = AutoConNet_Improved
