import torch
import torch.nn as nn

"""
8 并行化
    并行化包括数据并行化和模型并行化

    数据并行化比较简单，对模型进行指定即可
        model = model.cuda()
        model = nn.DataParallel(model,device_ids={0,1,2,3})
    模型并行化比较复杂，但是方便对大模型进行计算。

"""