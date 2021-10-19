import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict



"""
2 模块的深度学习模型构建方法
    下述为4中Module的继承模式，可以进行深度学习模型的搭建。
"""

"""
class Module(object):
    def __init__(self):
    def forward(self, *input):
 
    def add_module(self, name, module):
    def cuda(self, device=None):
    def cpu(self):
    def __call__(self, *input, **kwargs):
    def parameters(self, recurse=True):
    def named_parameters(self, prefix='', recurse=True):
    def children(self):
    def named_children(self):
    def modules(self):  
    def named_modules(self, memo=None, prefix=''):
    def train(self, mode=True):
    def eval(self):
    def zero_grad(self):
    def __repr__(self):
    def __dir__(self):
"""

"""
定义网络时，需要继承nn.module
并重新实现__init__和forward
- 一般把网络中具有可学习参数的层放在构造函数中，包括全连接层、卷积层等。
- 一般把不具有可学习参数的层，放到forward中，可以放在构造函数中，也可以不放在构造函数中。
- forward是必须要重写的，它实现模型的功能，实现各个层之间的连接。
"""

class mynet(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(mynet,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.relu1 = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=(2,2),stride=(1,1))
        self.conv2 = nn.Conv2d(3,32,3,1,1)
        self.relu2 = nn.ReLU()
        self.max_pooling2 = nn.MaxPool2d(2,1)
        self.dense1 = nn.Linear(32*3*3,128)
        self.dense2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling2(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

"""
上面将所有的层都放在构造函数中，在forward中实现了各个层之间的联系。
下面将没有学习参数的层，放在forward中，打印时，不会被打印出来。
在forward中实现，需要借助functional的函数进行实现。
"""

class mynet2(nn.Module):
    def __init__(self):
        super(mynet2, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,1,1)
        self.conv2 = nn.Conv2d(3,32,3,1,1)
        self.dense1 = nn.Linear(32 * 3 * 3, 128)
        self.dense2 = nn.Linear(128, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(2,1)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(2, 1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
"""
通过Sequential包装层

方式1
"""

class mynet3_1(nn.Module):
    def __init__(self):
        super(mynet3_1, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(3,32,1,1,1),nn.ReLU(),nn.MaxPool2d(2,1))
        self.dense_block = nn.Sequential(nn.Linear(32*3*3,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out


"""
self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 32, 3, 1, 1)),
                    ("relu1", nn.ReLU()),
                    ("pool", nn.MaxPool2d(2))
                ]
            ))

"""

class mynet3_2(nn.Module):
    def __init__(self):
        super(mynet3_2, self).__init__()
        self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    ("conv1",nn.Conv2d(3,32,1,1,1)),
                    ("relu1",nn.ReLU()),
                    ("max_pool",nn.MaxPool2d(2,1))
                ]
            ))
        self.dense_block = nn.Sequential(nn.Linear(32*3*3,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out
"""
self.conv_block.add_module("conv1",torch.nn.Conv2d(3, 32, 3, 1, 1))


"""
class mynet3_3(nn.Module):
    def __init__(self):
        super(mynet3_3, self).__init__()
        self.conv_block = nn.Sequential()
        self.conv_block.add_module("conv1",nn.Conv2d(3,32,1,1,1))
        self.conv_block.add_module("relu1",nn.ReLU())
        self.conv_block.add_module("max_pool",nn.MaxPool2d(2,1))
        self.dense_block = nn.Sequential(nn.Linear(32*3*3,128),nn.ReLU(),nn.Linear(128,10))
    def forward(self,x):
        conv_out = self.conv_block(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense_block(res)
        return out

if __name__ =='__main__':
    model1 = mynet()
    model2 = mynet2()
    model3_1 = mynet3_1()
    model3_2 = mynet3_2()
    model3_3 = mynet3_3()
    print(model1)
    print("\n")
    #print(model2)
    print("\n")
    #print(model3_1)
    print("\n")
    #print(model3_2)
    print("\n")
    #print(model3_3)
    """
    module.children
    实现迭代器
    """

    for i in model3_3.children():
        print(i)
        print(type(i))
    """
Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (max_pool): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
)
<class 'torch.nn.modules.container.Sequential'>
Sequential(
  (0): Linear(in_features=288, out_features=128, bias=True)
  (1): ReLU()
  (2): Linear(in_features=128, out_features=10, bias=True)
)
<class 'torch.nn.modules.container.Sequential'>
    """

    print("--------------------------------------------------")
    for i in model3_3.named_children():
        print(i)
        print(type(i))
"""
('conv_block', Sequential(
  (conv1): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU()
  (max_pool): MaxPool2d(kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=False)
))
<class 'tuple'>
('dense_block', Sequential(
  (0): Linear(in_features=288, out_features=128, bias=True)
  (1): ReLU()
  (2): Linear(in_features=128, out_features=10, bias=True)
))
<class 'tuple'>
"""

'''
总结：
named_children 返回的数据会携带名称
'''