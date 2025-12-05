from torch import nn

import torch


class Transformer(nn.Module):
    def __init__(self, num_encoder: int=5, num_decoder:int =5, ):
        super().__init__()
        self.num_encoder = num_encoder
        self.num_decoder = num_decoder
        self.embed
        self.PE
        self.multi_head_attention
        self.add_norm
        self.fnn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        pass
    