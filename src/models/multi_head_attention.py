import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()

        assert embedding_dim % num_heads == 0

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Получаем Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 2. Разделяем embedding на головы
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. Ставим heads перед seq_len
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. Считаем оценки внимания
        scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # 5. Causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # 6. Превращаем scores в вероятности внимания
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # 7. Взвешиваем Value
        output = attention_weights @ V

        # 8. Возвращаем головы обратно в одну размерность
        output = output.transpose(1, 2)

        output = output.contiguous().view(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        # 9. Финальный Linear
        output = self.output(output)

        return output