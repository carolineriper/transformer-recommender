import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    seen_histories: dict[int, list[int]],
    k: int = 10,
) -> tuple[float, float]:
    """Compute HR@K and NDCG@K after masking each user's full history."""
    if k <= 0:
        raise ValueError("k must be a positive integer")

    model.eval()

    total_hits = 0
    total_ndcg = 0.0
    total_users = 0

    with torch.inference_mode():
        for x_batch, y_batch, user_ids in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)

            if logits.ndim != 2:
                raise ValueError("model output must have shape [batch, num_movies]")
            if k > logits.size(1):
                raise ValueError("k cannot exceed the number of candidate movies")

            mask_value = torch.finfo(logits.dtype).min

            for i, user_id in enumerate(user_ids.tolist()):
                seen_movies = seen_histories[user_id]

                if seen_movies:
                    seen_indices = torch.tensor(
                        seen_movies,
                        dtype=torch.long,
                        device=device,
                    )
                    logits[i, seen_indices] = mask_value

            topk = torch.topk(
                logits,
                k=k,
                dim=1,
            ).indices

            matches = topk == y_batch.unsqueeze(1)

            hits = matches.any(dim=1)
            total_hits += hits.sum().item()

            for match in matches:
                positions = torch.where(match)[0]
                if len(positions) > 0:
                    rank = positions[0].item() + 1
                    total_ndcg += 1 / math.log2(rank + 1)

            total_users += y_batch.size(0)

    if total_users == 0:
        raise ValueError("data_loader produced no evaluation examples")

    return total_hits / total_users, total_ndcg / total_users
