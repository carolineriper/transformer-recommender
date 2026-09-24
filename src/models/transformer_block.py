import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
        )

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x