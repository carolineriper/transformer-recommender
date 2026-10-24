
import math
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_topk_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    seen_histories: dict[int, list[int]],
    k: int = 10,
) -> tuple[list[list[int]], list[int]]:
    """Получить Top-K рекомендации и правильные ответы."""

    model.eval()
    recommendations = []
    targets = []

    with torch.inference_mode():
        for x_batch, y_batch, user_ids in data_loader:
            logits = model(x_batch.to(device))

            if not 1 <= k <= logits.shape[1]:
                raise ValueError("Invalid k")

            # Исключаем фильмы, уже просмотренные пользователем.
            for i, user_id in enumerate(user_ids.tolist()):
                seen = seen_histories[user_id]
                if seen:
                    logits[i, seen] = torch.finfo(logits.dtype).min

            topk = torch.topk(logits, k=k, dim=1).indices

            recommendations.extend(topk.cpu().tolist())
            targets.extend(y_batch.tolist())

    if not targets:
        raise ValueError("No evaluation examples")

    return recommendations, targets


def hit_rate_at_k(
    recommendations: list[list[int]],
    targets: list[int],
) -> float:
    """Доля пользователей, у которых target попал в Top-K."""

    hits = sum(
        target in recs
        for recs, target in zip(recommendations, targets)
    )
    return hits / len(targets)


def ndcg_at_k(
    recommendations: list[list[int]],
    targets: list[int],
) -> float:
    """
    Средний NDCG при одном релевантном фильме на пользователя.
    Наказывает за высокую позицию нерелевантного объекта и 
    За невысокую позицию релевантного объекта.
    Так как известен только 1 следующий фильм в истории, 
    То IDCG всегда = 1, если с фильмом угадали и = 0, если не угадали.  
    """

    total = 0.0

    for recs, target in zip(recommendations, targets):
        if target in recs:
            rank = recs.index(target) + 1
            total += 1 / math.log2(rank + 1)

    return total / len(targets)


def catalog_coverage_at_k(
    recommendations: list[list[int]],
    num_movies: int,
) -> float:
    """
    Доля уникальных рекомендованных фильмов во всём каталоге.
    Складываем количество уникальных показанных фильмов 
    И делим на общее количество имеющихся фильмов.
    """

    unique_movies = set()

    for recs in recommendations:
        unique_movies.update(recs)

    return len(unique_movies) / num_movies


def novelty_at_k(
    recommendations: list[list[int]],
    train_histories: dict[int, list[int]],
    num_movies: int,
) -> float:
    """Средняя новизна рекомендаций через -log2(popularity)
    Считаем популярность каждого фильма по просмотрам (нормируя Лапласом)
    И превращаем в новизну через логарифм.
    """

    counts = Counter(
        movie
        for history in train_histories.values()
        for movie in history
    )

    total_interactions = sum(counts.values())
    novelty_scores = []

    for recs in recommendations:
        for movie in recs:
            probability = (
                (counts[movie] + 1)
                / (total_interactions + num_movies)
            )
            novelty_scores.append(-math.log2(probability))

    return sum(novelty_scores) / len(novelty_scores)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    seen_histories: dict[int, list[int]],
    k: int = 10,
) -> tuple[float, float]:
    """Совместимость с существующим циклом обучения."""

    recommendations, targets = get_topk_predictions(
        model, data_loader, device, seen_histories, k
    )

    return (
        hit_rate_at_k(recommendations, targets),
        ndcg_at_k(recommendations, targets),
    )
