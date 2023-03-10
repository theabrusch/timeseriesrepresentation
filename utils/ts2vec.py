import torch
from torch import nn
import numpy as np

class DilatedCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size):
        super().__init__()
        padding = ((kernel_size - 1) * dilation + 1)//2
        self.layer = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding = padding),
            nn.GELU(),
            nn.Conv1d(in_channels=out_channels,out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding = padding)
        )
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.projector is None else self.projector(x)
        out = self.layer(x)
        return res + out
    
def generate_binomial_mask(B, T, p = 0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class TS2VecEncoder(nn.Module):
    def __init__(self, input_size, hidden_channels, out_dim, nlayers = 10, kernel_size = 3):
        super().__init__()
        self.linear_projection = nn.Linear(in_features=input_size, out_features=hidden_channels)

        in_channels = [hidden_channels]*(nlayers + 1)
        out_channels = [hidden_channels]*nlayers + out_dim
        dilation = [2**i for i in range(nlayers)]
        convblocks = [DilatedCNNBlock(in_ch, out_ch, dil, kernel_size) for in_ch, out_ch, dil in zip(in_channels, out_channels, dilation)]
        self.convblocks = nn.Sequential(*convblocks)
    
    def forward(self, x):
        proj = self.linear_projection(x)
        mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)




