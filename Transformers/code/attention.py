import torch
import torch.functional as F 


def attention(Q, K, V):
    d_k = Q.size(-1)
    attention_score = torch.matmul(F.softmax(torch.matmul(Q, K.T) / torch.sqrt(d_k)), V)
    return attention_score

def self_attention(X):
    return attention(X, X, X)

def mask_self_attention(X):
    d = X.shape[-1]
    mask = torch.triu(torch.full(size=(1, d, d), dtype=float("-inf")))
    score = F.softmax(self_attention(X) + mask, dim=-1)
    return score


class MutiHeadAttention(nn.Module):
    def __init__(self, batch_size, seq_len, dim):
        super().__init__()
        

    def forward(self, x):
        pass
    