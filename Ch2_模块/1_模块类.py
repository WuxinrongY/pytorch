import torch
from torch.functional import Tensor
import torch.nn as nn
"""
1 模块类
模块本身是一个类nn.Module
torch的模型通过继承该类，在类的内部定义模块的实例化，通过前向计算调用模块，最后实现深度学习模型的搭建。
    1.1 简单线性回归类
    1.2 模块函数的使用
"""

class ModelTest(nn.Module):
    # 定义初始化函数,x是用户传入的参数
    def __init__(self,x):
        super().__init__()

    def forward(self,x):
        ret = x
        return ret

# 简单线性回归类

class LineraModel(nn.Module):
    def __init__(self,ndim):
        super(LineraModel,self).__init__()
        self.ndim = ndim
        # 定义权重
        self.weight = nn.Parameter(torch.randn(ndim,1))
        # 定义偏置
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self,x):
        # y = wx+b
        return x.mm(self.weight)+self.bias

model = LineraModel(2)
"""
此处出现报错，原因，当将python数组转换为tensor时，1x2为二维数组，而2为一维数组，因此，需要[[1,2]]才能表示1x2的数组
"""
T1 = torch.tensor([[1,2]],dtype=torch.float)
T = torch.randn(2,2)
print(T.dtype)
print(T1.dtype)
print(model(T1))
Pa = list(model.named_parameters())
print(Pa)
# 转换模型参数为半精度浮点数
model.half()
Pa = list(model.named_parameters())
print(Pa)
# model.cuda() 将模型转移到cuda上




    