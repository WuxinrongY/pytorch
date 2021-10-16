import torch
"""
获取维度的数目
torch.ndimension()
获取该张量的总的元素数目
torch.nelement()
获取张量的大小
torch.size()
获取张量维度0的大小
torch.size(0)
变更张量形状
torch.view(3,3)
变更张量的形状，第一维度自动计算
torch.view(-1,3)
获取张量的数据指针
torch.data_ptr()
交换两个维度的步长
torch.transpose()
"""

T = torch.randn(3,3)
print(T.ndimension())
print(T.nelement())
print(T.size())
print(T.size(0))
# view会对原有的张量进行重置大小，但是要注意，指定的维度可容纳的元素要和之前一致。
T = T.view(1,9)
print(T)
T = T.transpose(0,1)
print(T)
print(T.data_ptr())
T.reshape(10,2)
print(T)