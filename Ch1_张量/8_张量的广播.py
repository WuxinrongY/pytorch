import torch

"""
7 张量的广播

    广播的目的是允许不同大小的张量进行四则运算
    但是最多允许1个维度大小不同
"""

T1 = torch.randn(3,4)
T2 = torch.randn(3,4,2)

T1 = T1.unsqueeze(-1)

print(T1.shape)
print(T2.shape)
print(T1+T2)