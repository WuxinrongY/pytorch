import torch
import torch.nn as nn
from torch.nn.modules.loss import BCELoss
"""
4 损失函数
    二分类交叉熵损失函数：该函数主要用于处理sigmoid的函数输出，进行二分类，但是该损失函数在输入接近1时，可能出小溢出等数值不稳定的情况，因此，采用对数损失函数进行缓解。
    对数交叉熵损失函数

    负对数似然函数：根据预测值（经过softmax的计算和对数计算）和目标值计算这两个值安装元素一对一的对应的乘积，然后对乘积求和，并取负值。
    因此，该函数主要处理softmax的计算结果（需要取对数后才能作为输入）    torch.nn.functional.log_softmax可以实现这个功能。
"""


# 初始化平方损失函数模块
mse = nn.MSELoss()

t1 = torch.randn(5,requires_grad=True)
t2 = torch.randn(5,requires_grad=True)
t = mse(t1,t2)

sum = t1.sub(t2).pow(2).sum()

print(t1)
print(t2)
print(t)
print(sum)


t1s = t1.sigmoid()
print(t1s)
t2 = torch.randint(0,2,(5,)).float()

bce = nn.BCELoss()
t = bce(t1s,t2)
print(t1s)

print(t)

bce_log = nn.BCEWithLogitsLoss()

t = bce_log(t1s,t2)
print(t)

