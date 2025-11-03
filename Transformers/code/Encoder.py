from torch import nn

import torch


class MLP(nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_ch, hid_ch),
            nn.ReLU(),
            nn.Linear(hid_ch, out_ch),
            nn.Dropout()
        )

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  
        self.beta = nn.Parameter(torch.zeros(d_model))  
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = LayerNorm()
        self.attention 
        self.fnn
        
    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


if __name__ == '__main__':
    mlp = MLP(3, 6, 3)
    print(mlp)