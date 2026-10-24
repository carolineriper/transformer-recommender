
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def build_genre_matrix(
    movies: pd.DataFrame,
    movie_ids: list[int],
) -> tuple[np.ndarray, list[str]]:

    genres = (
        movies.set_index("movie_id") # делаем ID фильма индексом таблицы
        .loc[movie_ids, "genres"] # выбираем жанры только тех 3706 фильмов, который используются в трансформере
        .str.split("|") # превращает строку "Action|Sci-Fi" в список ["Action", "Sci-Fi"]
    )

    encoder = MultiLabelBinarizer() # преобразует списки жанров в Multi-Hot векторы
    genre_matrix = encoder.fit_transform(genres)

    return (
        genre_matrix.astype(np.float32),
        encoder.classes_.tolist(), # сохраняет названия жанров, чтобы мы понимали, какому признаку соответствует каждый столбец.
    )


def build_user_profile(
    user_id: int,
    train_ratings: pd.DataFrame,
    genre_matrix: np.ndarray,
    gamma: float = 1.0 # Коэффициент затухания
) -> np.ndarray:

    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1]')
    
    # Получаем оценки фильмов одного конкретного пользователя
    user_data = train_ratings[
        train_ratings["user_id"] == user_id
    ] 
    # Получаем индексы просмотренных фильмов
    movie_indices = user_data["movie_idx"].to_numpy()

    # Превращаем оценки в веса (-3, чтобы для оценки 3 вес был нулевой как для медианы)
    rating_weights = (
        user_data['rating'].to_numpy(dtype=np.float32) - 3
    )

    # Коэффициенты временного затухания
    n = len(rating_weights)
    decay = gamma ** np.arange(n - 1, -1, -1)

    # Объединяем оценку и давность взаимодействия
    weights = rating_weights * decay

    # Достаем жанры просмотренных фильмов
    movie_genres = genre_matrix[movie_indices]

    # Умножаем вектор жанров каждого фильма на его вес и складываем результаты
    # То есть получается вектор жанров учитывая их веса (на основе оценки)
    weighted_genres = weights @ movie_genres

    # Считаем сумму модулей весов, чтобы потом нормализовать профиль пользователя
    # Так мы уменьшим зависимость профиля пользователя от количества оцененных фильмов
    denominator = np.abs(weights).sum()

    if denominator == 0:
        return np.zeros(genre_matrix.shape[1], dtype=np.float32)

    return weighted_genres / denominator

def recommend_content_based(
    user_id: int,
    train_ratings: pd.DataFrame,
    genre_matrix: np.ndarray,
    popularity_scores: np.ndarray,
    k: int = 10,
    gamma: float = 1.0,
) -> list[int]:

    # Строим профиль пользователя
    user_profile = build_user_profile(
        user_id=user_id,
        train_ratings=train_ratings,
        genre_matrix=genre_matrix,
        gamma=gamma
    )

    # Если профиль нулевой, используем популярность
    if np.linalg.norm(user_profile) < 1e-8:
        scores = popularity_scores.astype(np.float64).copy()

    else:
        # В обычном случае используем Cosine Similarity
        scores = cosine_similarity(
            user_profile.reshape(1, -1),
            genre_matrix,
        )[0]

    # Получаем известные фильмы пользователя
    # В train_ratings содержатся user_id - movie_idx - ratings
    user_data = train_ratings[
        train_ratings["user_id"] == user_id
    ]
    seen_movies = user_data["movie_idx"].to_numpy()

    # Исключаем известные фильмы
    scores[seen_movies] = -np.inf

    # Выбираем доступные Top-K
    n_available = len(scores) - len(np.unique(seen_movies))
    # np.lexsort() сортирует по нескольким критериям
    topk = np.lexsort((popularity_scores, scores))[::-1][:min(k, n_available)]

    return topk.tolist()





