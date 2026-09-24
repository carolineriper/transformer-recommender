import torch
import torch.nn as nn


class MovieEmbedding(nn.Module):
    def __init__(
        self,
        num_movies: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_movies,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        movie_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embedding(movie_ids)