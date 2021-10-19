import torch
from torch.autograd import grad

"""
3 计算图和自动求导机制
    自动求导机制中，创建向量时需要关注参数, requires_grad = True
    意味着这个张量会加入到计算图中，作为计算图的叶子节点参与计算。
    一旦设定了该参数，后续得到的中间结果也会和它一致。
"""
T1 = torch.randn(3,3,requires_grad=True)
T2 = torch.randn(3,3,requires_grad=False)
print(T1)
print(T2)
# 计算张量的所有分量的平方和
T = T1.pow(2).sum()
# 反向传播
T.backward()
# T1的梯度信息被改变
print(T1.grad)

T = T2.pow(2).sum()

# T.backward() 此处报错，不可执行
print(T2.grad)

T = T1.pow(2).sum()
grad = torch.autograd.grad(T,T1)# T对T1求导数，但是不会反向传播。
print(grad)

print(T1.grad.zero_())# 张量的梯度清零


"""
计算图的启用和禁用
"""
print("--------------------------------------")
t1 = torch.randn(3,3,requires_grad=True)
t2 = t1.sum()
# t2的输出结果带有grad_fn 说明t2的计算构建了计算图
print(t2)

with torch.no_grad():
    t3 = t1.sum()
    print(t3)
# detach会生成新的张量，与旧的张量分离，互不影响。
t4 = t1.sum().detach()

print(t4)
