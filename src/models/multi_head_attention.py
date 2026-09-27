import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with optional padding masking."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

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

        self.output = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        queries = queries.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        keys = keys.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        values = values.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = (
            queries @ keys.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(
            ~causal_mask,
            mask_value,
        )

        if padding_mask is not None:
            key_padding_mask = padding_mask[:, None, None, :]
            scores = scores.masked_fill(
                key_padding_mask,
                mask_value,
            )

        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ values

        if padding_mask is not None:
            query_padding_mask = padding_mask[:, None, :, None]
            output = output.masked_fill(
                query_padding_mask,
                0.0,
            )

        output = output.transpose(1, 2)
        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim,
        )
        output = self.output(output)

        if padding_mask is not None:
            output = output.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )

        return output
