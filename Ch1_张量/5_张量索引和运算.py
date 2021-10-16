import torch

"""
5 张量的索引和运算
    5.1 张量的索引
        ：表示全部，0：3表示x>=0 && x<3
        T>0表示对全部的元素进行运算
    5.2 张量的运算
"""

# 张量的索引
T = torch.randn(2,5,3)
print(T[1,1,1])

print(T[:,0:3,1])
print(T)
print(T[T>0])

# 张量的运算

