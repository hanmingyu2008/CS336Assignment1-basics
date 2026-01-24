import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device | None = None, dtype:torch.dtype | None = None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        std = 1
        nn.init.trunc_normal_(self.embedding_matrix, mean = 0, std = std, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:

        return self.embedding_matrix[token_ids]