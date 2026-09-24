import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = (
            Q @ K.transpose(-2, -1) # меняем последние две оси
        ) / math.sqrt(Q.size(-1)) # последняя размерность - embedding_dim = 128

        seq_len = x.size(-2)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        output = attention_weights @ V

        return output