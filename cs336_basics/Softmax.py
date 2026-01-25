import torch
from torch import nn

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:

    # x.max()返回的会是元组(values, indices),所以需要提取values一项
    x_max = x.max(dim = i, keepdim=True)[0]
    
    out = torch.exp(x-x_max)

    return out / torch.sum(out, dim = i, keepdim=True)