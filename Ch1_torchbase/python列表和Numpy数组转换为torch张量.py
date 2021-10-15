
import torch
import numpy as np
print(torch.__version__)
"""
1 python列表和Numpy数组转换为pytorch张量
1.1 tensor类型
------
32位浮点数
tensor.float
tensor.float32
------
16位浮点数
tensor.float16
------
64位浮点数
tensor.float64
tensor.double
------
8位无符号数
tensor.uint8
------
16位带符号整数
tensor.int16
tensor.short
------
32位带符号整数
tensor.int32
tensor.int
------
布尔
tensor.bool

"""
# ----------------------
# 转换python为torch张量
T = torch.tensor([1,2,3,4])
print(T)
T = torch.tensor([1.1,2,3,4])
print(T.dtype)
T = torch.tensor([1,2,3,4],dtype=torch.float)
print(T.dtype)
# ----------------------
# 转换迭代器为张量
T = torch.tensor(range(10))
print(T)

# ----------------------
# numpy转换为torch张量
n = np.array([1,2,3,4])
print(n.dtype)
print(n)

T = torch.tensor(n)
print(T)

# torch默认的浮点类型是float32，numpy默认的浮点类型为double,因此转换后，tensor的类型为float64
n = np.array([1,2,3,4.1])
print(n.dtype)
print(n)

T = torch.tensor(n)
print(T)

# ----------------------
# 列表套用转换为torch张量
T = torch.tensor([[1,2,3],[3,4,5],[6,7,8]])
print(T)

# ----------------------
# torch类型转换
T = torch.randn(3,3,3)
print(T)
print(T.dtype)

T = torch.randn(3,3,3).to(torch.int)
print(T)
print(T.dtype)


