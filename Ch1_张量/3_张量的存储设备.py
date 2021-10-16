import torch
"""
3. 张量的存储设备
"""
# 张量存储在cpu
T = torch.randn(3,3,device="cpu")
print(T)
print(T.device)
# T = torch.randn(3,3,device="cuda:0")
# print(T)

# 张量转移
# T.cuda(0)
# T.to("cuda:0")
