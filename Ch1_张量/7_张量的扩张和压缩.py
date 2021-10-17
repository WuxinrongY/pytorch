import torch
"""
torch.unsqueeze(),扩展维度,维度+1
squeeze 压缩所有大小为1 的维度

"""

T = torch.randn(3,4)
T1 = T.unsqueeze(-1)
print(T1.shape)
print(T1.squeeze().shape)