import torch
import torch.nn as nn

from .embedding import MovieEmbedding
from .positional_embedding import PositionalEmbedding
from .transformer_block import TransformerBlock


class TransformerRecommender(nn.Module):
    """Causal Transformer for full-catalog next-movie prediction."""

    def __init__(
        self,
        num_movies: int,
        embedding_dim: int,
        max_seq_len: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.padding_idx = num_movies

        self.movie_embedding = MovieEmbedding(
            num_movies=num_movies,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        self.output = nn.Linear(
            embedding_dim,
            num_movies,
        )

    def forward(
        self,
        movie_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, seq_len = movie_ids.shape

        padding_mask = movie_ids == self.padding_idx

        positions = torch.arange(
            seq_len,
            device=movie_ids.device,
        )

        movie_vectors = self.movie_embedding(movie_ids)
        position_vectors = self.position_embedding(positions)
        x = movie_vectors + position_vectors

        for block in self.blocks:
            x = block(
                x,
                padding_mask=padding_mask,
            )

        x = self.norm(x)
        x = x[:, -1, :]
        return self.output(x)
