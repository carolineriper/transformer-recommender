import torch
import torch.nn as nn


class MovieEmbedding(nn.Module):
    """Learn movie representations with a dedicated padding index."""

    def __init__(
        self,
        num_movies: int,
        embedding_dim: int,
        padding_idx: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_movies + 1,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, movie_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(movie_ids)
