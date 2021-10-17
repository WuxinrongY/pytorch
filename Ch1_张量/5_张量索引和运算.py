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
T = torch.tensor([1,2,3],dtype=float)
print(T)
print(T.sqrt())
print(T)
# 上方实验证明，尽管是直接对T进行开方，但是并不会对原始数据造成影响
# sqrt_是原地平方操作 使用时需要注意数据类型
print(T.sqrt_())
print(T)

print(T.sum())

T1 = torch.randn(2,3)
T2 = torch.randn(2,3)
T = T1.add(T2)
print(T1)
print(T2)
print(T)

T = T1.sub(T2)
print(T)


# 矩阵乘法
"""
torch.mm(a,b)
a.mm(b)
a@b

a = 2*3*4
b = 2*4*3
a.bmm(b) = 2*3*3
a.einsum("bnk,bkl->bnl",a,b)# 此函数中，不同的字母代表不同的维度的大小
"""

