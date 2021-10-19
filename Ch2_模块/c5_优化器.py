import torch
import torch.nn as nn
from sklearn.datasets import load_boston
from c1_模块类 import *

"""
实验内容：
    1 使用c1_模块类中的线性回归模型，构建有13个参数的线性回归模型。
    2 构建损失函数计算模块critersion，并将其设置为MSELoss
    3 构建随机梯度下降算法(torch.optim.SGD)，优化器的第一个参数是线性回归模型参数的生成器，第二个参数是学习率
    4 获取数据集
    5 获取当前模型的预测结果
    6 将获取的预测结果与目标进行损失计算
    7 清空梯度，因为多次计算会使梯度累计
    8 损失函数反向传播，计算得到每个参数对应的梯度
    9 执行一步优化的计算。
"""

boston = load_boston()
lm = LineraModel(13)
criterion = nn.MSELoss()
# 定义优化器
optim = torch.optim.SGD(lm.parameters(),lr=1e-6)
data = torch.tensor(boston["data"], requires_grad=True,dtype=torch.float32)
target = torch.tensor(boston["target"],dtype=torch.float32)


if __name__ == '__main__':
    for step in range(20000):
        predict = lm(data)
        loss = criterion(predict,target)
        if step and step %1000 == 0:
            print("loss:{:.3f}".format(loss.item()))
        optim.zero_grad()
        loss.backward()
        optim.step()


