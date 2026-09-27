import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learn an embedding for each position in the fixed-length context."""

    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(positions)
