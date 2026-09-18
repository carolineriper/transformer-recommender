from pathlib import Path
import pandas as pd


def load_movielens(data_dir: Path):
    """
    Load MovieLens 1M ratings and movies tables.

    Parameters
    ----------
    data_dir : Path
        Path to the MovieLens directory.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Ratings and movies DataFrames.
    """
        
    ratings_path = Path(data_dir) / 'ratings.dat' 

    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )

    movies_path = Path(data_dir) / 'movies.dat'
    movies = pd.read_csv(
        movies_path,
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    

    return ratings, movies

if __name__ == '__main__':
    ratings, movies = load_movielens(Path('data/raw/ml-1m'))
    print(ratings.head())
    print()
    print(movies.head())