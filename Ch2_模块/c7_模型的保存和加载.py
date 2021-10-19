import torch
from torch.serialization import save
from c1_模块类 import *
"""
7 模型的保存和加载

    建议使用状态字典来保存模型，可以同时保存多个模型。
    
    在torch中，保存模型分两种，一种是直接保存模型实例，一种是保存模型的状态字典。
        
    模型和张量的序列化
        torch.save(obj，f，pickle，pickle_protocol=2)
            第一个参数是模型或者张量，第二个是路径，第三个是序列化方法，第4个是序列化方法的标准，目前为0-4号

        torch.load(f,map_location,pickle_module = pickle)
            第一个参数是路径，第二个参数是张量存储位置的映射，如果保存时，模型或者张量就在CPU中，使用默认参数没问题，但是当保存时，张量存储在GPU中，但是load的系统无GPU，会出现无法Load的情况
            可以使用map_loaction="cpu"进行操作。 因为读取模型时，先读取到CPU，再转移到GPU，出错会在转移时出现。
    状态字典的保存和加载
        model.state_dict()
            获取模型的状态字典

        model.load_state_dicy()
            加载模型的状态字典
"""
lm = LineraModel(5)
lmdic = lm.state_dict()

# 保存
# 直接保存模型测试失败
# torch.save(lm,"线性回归模型.pth")

save_info = {
    "item_num":5,
    "model":lm.state_dict()
}
torch.save(save_info,"线性回归模型状态字典.dic")

print(lmdic)


# 加载

load_info = torch.load("线性回归模型状态字典.dic")
print(load_info)

lm1 = LineraModel(5)
for p in lm1.parameters():
    print(p)

lm1.load_state_dict(load_info["model"])
for p in lm1.parameters():
    print(p)

