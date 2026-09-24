import torch
import torch.nn as nn

from .embedding import MovieEmbedding
from .positional_embedding import PositionalEmbedding
from .transformer_block import TransformerBlock


class TransformerRecommender(nn.Module):
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

        # Embeddings
        self.movie_embedding = MovieEmbedding(
            num_movies=num_movies,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embedding_dim)

        # 128 -> num_movies
        self.output = nn.Linear(
            embedding_dim,
            num_movies,
        )