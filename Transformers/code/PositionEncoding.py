import torch
import numpy as np


def get_pos_encode(X: torch.Tensor) -> torch.Tensor:
    '''
    返回X的位置编码矩阵，注意X应为2维，返回后仍需与X相加才能得到正确的矩阵
    '''
    seq_len, dim = X.size(0), X.size(1)
    
    P = torch.zeros_like(X)
    for row in range(seq_len):
        for line in range(dim):
            P[row, line] = _pos_encode(row, line, dim)
    
    return P

def _pos_encode(pos, i, dim):
    if i == 1:
        return np.cos(pos / np.power(10000, (i-1)/dim))
    if i % 2 == 0:
        return np.sin(pos / np.power(10000, i/dim))
    else:
        return np.cos(pos / np.power(10000, (i-1)/dim))


if __name__ == '__main__':
    X = torch.Tensor([[0.1, 0.2, 0.3, 0.4],
                      [0.2, 0.3, 0.4, 0.5],
                      [0.3, 0.4, 0.5, 0.6],
                      [0.4, 0.5, 0.6, 0.7]])
    P = get_pos_encode(X)
    print(X + P)