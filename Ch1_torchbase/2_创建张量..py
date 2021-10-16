import torch
"""
2 创建张量
    2.1 通过torch.tensor进行转换
        具体内容可见上节
    2.2 通过内置函数创建
        torch.randn()
        torch.zeros()
        torch.ones()
        torch.eye()
        torch.randint()
    2.3 通过已知张量创建相同形状的张量
        torch.zeros_like()
        torch.ones_like()
        torch.rand_like()
        torch.randn_like()
    2.4 通过已知的张量生产类型不同的新的张量
        创建的张量形状不同，但是数据类型相同
        t.new_zeros()
"""
# 2.2 通过内置函数创建
# randn()中可以写数子表示该维度的大小。如randn(3,3,3,4)表示生成向量前3维是3，第4维度是4
T = torch.randn(1,2,3,4)
print(T)
T = torch.zeros(1,2,3,4)
print(T)
T = torch.ones(1,2,3,4)
print(T)
T = torch.eye(3)
print(T)
T = torch.randint(1,10,(3,3,3))# randint(low,high,(size))
print(T)

# 2.3 通过已知张量创建相同形状的张量
T = torch.randn(3,3,3)
T2 = torch.zeros_like(T)
print(T2)
T2 = torch.ones_like(T)
print(T2)
T2 = torch.rand_like(T)
print(T2)
T2 = torch.randn_like(T)
print(T2)

# 2.4 通过已知的张量生产类型不同的新的张量
T3 = T.new_ones(3,3)
print(T3)
print(T)
print(T3.dtype)
print(T.dtype)


