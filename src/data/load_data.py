from pathlib import Path

import pandas as pd


def load_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens 1M ratings and movie metadata tables."""
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ratings.dat"
    movies_path = data_dir / "movies.dat"

    missing_files = [
        path.name
        for path in (ratings_path, movies_path)
        if not path.is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing MovieLens file(s) in {data_dir}: {missing}. "
            "Download MovieLens 1M and extract it to data/raw/ml-1m/."
        )

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    return ratings, movies
