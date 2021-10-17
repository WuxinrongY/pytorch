import torch

"""
6 张量的拼接和分割
    torch.stack()# 将传入的张量堆叠起来，前提是这些张量的大小必须一致。
    torch.cat()# 将传入的张量沿着指定的维度拼接， 除了要拼接的维度外，其他维度必须保持一致
    torch.split()# 将张量在指定维度进行分割
    torch.chunk()# 与上述函数功能类似
"""

t1 = torch.randn(3,4)
t2 = torch.randn(3,4)
t3 = torch.randn(3,4)
t4 = torch.randn(3,4)
# 多个二维向量堆叠后，变为3维向量
T = torch.stack((t1,t2,t3),-1)
print(T.shape)

# 张量拼接
T = torch.cat((t1,t2,t3),-1)
print(T.shape)

# 分割T
T1 = torch.split(T,(3,5,4),-1)
print(T1[0].shape)
print(T1[1].shape)
print(T1[2].shape)

# chunk最后一个维度平均分为3分，每份多大自动计算
T1 = torch.chunk(T,3,-1)
print(T1[0].shape)
print(T1[1].shape)
print(T1[2].shape)


