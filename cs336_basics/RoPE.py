import torch
import torch.nn as nn

class RoPE(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta 
        self.d_k = d_k 
        self.max_seq_len = max_seq_len
        self.device = device
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        # 
        positions = torch.arange(self.max_seq_len)
        # 
        sinusoids = torch.outer(positions, freqs) 
        # register_buffer: 不可训练, nn.Parameter(): 可训练
        # persisent = True: 保存, False: 不保存
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 这里是register_buffer提取的过程
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        cos = cos.unsqueeze(0) # shape: [1, max_seq_len, d_k//2] 对应 [batch, max_seq_len, d_k//2]
        sin = sin.unsqueeze(0) # shape: [1, max_seq_len, d_k//2] 对应 [batch, max_seq_len, d_k//2]

        x_part1 = x[..., 0::2]
        x_part2 = x[..., 1::2]

        output1 = x_part1 * cos - x_part2 * sin 
        output2 = x_part1 * sin + x_part2 * cos 

        out = torch.stack([output1, output2], dim=-1)  # [batch, seq_len, d_k//2, 2] #用stack能巧妙的把奇数和偶数交叉在一起，cat就不行
        out = out.flatten(-2)  # [batch, seq_len, d_k]
        return out