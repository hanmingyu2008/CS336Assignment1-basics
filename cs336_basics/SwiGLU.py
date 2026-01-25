import torch
from torch import nn
from .Linear import Linear

class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear(d_model,d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)

    def Silu(self, x):
        return x * torch.sigmoid(x)     # * 符号就是逐项乘，真正的张量乘是 @

    def forward(self, x):
        return self.W2(self.Silu(self.W1(x)) * self.W3(x))