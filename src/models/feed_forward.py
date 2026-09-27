import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network used in each Transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
