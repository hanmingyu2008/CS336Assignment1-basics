import torch
import torch.nn as nn
from .RoPE import RoPE
from .ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int ,device=None,rope_required:bool | None=False, theta:float | None = None, max_seq_len:int | None=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_required = rope_required
        
        if rope_required:
            self.theta = theta
            self.rope = RoPE(theta, self.head_dim, max_seq_len,device) #这里要注意，我们使用多头注意力机制的时候，每个head的维度是d_model // n_heads，我们应当对每个head进行RoPE
            self.max_seq_len = max_seq_len

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9) # 先前都为mask==0,点解突然变为1?
        attn_weights = torch.softmax(scores, -1) 
        return torch.matmul(attn_weights, V) 
    def forward(self, x, wq, wk, wv, wo, token_positions:torch.Tensor | None = None)->torch.Tensor:
        if self.rope_required and (token_positions == None):
            raise Exception("Need token_position input for rope_required Attention !")
        batch_size, seq_len, d_model = x.shape
        
        # x = self.rope(x, token_positions)
        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim) #view会优先切分最后一个维度，这和内存有关。
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.rope_required:
            q = self.rope(q, token_positions)     
            k = self.rope(k, token_positions)

        mask = torch.triu(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, d_model)
        out = out @ wo.T
        return out