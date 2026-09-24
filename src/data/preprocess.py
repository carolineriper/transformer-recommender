import pandas as pd


def sort_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Sort interactions by user and timestamp.

    Parameters
    ----------
    ratings : pd.DataFrame
        MovieLens ratings table.

    Returns
    -------
    pd.DataFrame
        Ratings sorted by user_id and timestamp.
    """
    sorted_ratings = ratings.sort_values(
        by=["user_id", "timestamp"],
        ignore_index=True,
    )

    return sorted_ratings


def build_user_histories(
    ratings: pd.DataFrame,
) -> dict[int, list[int]]:
    """
    Build ordered interaction histories for each user.

    Parameters
    ----------
    ratings : pd.DataFrame
        Ratings sorted by user_id and timestamp.

    Returns
    -------
    dict[int, list[int]]
        Mapping:
        user_id -> ordered list of movie_id values.
    """
    user_histories = (
        ratings
        .groupby("user_id")["movie_id"]
        .apply(list)
        .to_dict()
    )

    return user_histories


def temporal_split(
    user_histories: dict[int, list[int]],
) -> tuple[
    dict[int, list[int]],
    dict[int, int],
    dict[int, int],
]:
    """
    Split each user's history into train, validation and test.

    Train:
        All interactions except the last two.

    Validation:
        The second-to-last interaction.

    Test:
        The last interaction.

    Parameters
    ----------
    user_histories : dict[int, list[int]]
        Ordered interaction histories.

    Returns
    -------
    tuple
        train_histories:
            user_id -> train sequence

        val_targets:
            user_id -> validation item

        test_targets:
            user_id -> test item
    """
    train_histories = {}
    val_targets = {}
    test_targets = {}

    for user_id, history in user_histories.items():

        if len(history) < 3:
            continue

        train_histories[user_id] = history[:-2]
        val_targets[user_id] = history[-2]
        test_targets[user_id] = history[-1]

    return train_histories, val_targets, test_targets


def generate_sequences(
    train_histories: dict[int, list[int]],
    max_seq_len: int,
) -> list[tuple[int, list[int], int]]:
    """
    Generate next-item prediction examples using a sliding window.

    Each example has the form:

        (user_id, input_sequence, target_item)

    Parameters
    ----------
    train_histories : dict[int, list[int]]
        Training interaction histories.

    max_seq_len : int
        Maximum input sequence length.

    Returns
    -------
    list[tuple[int, list[int], int]]
        Training examples.
    """
    sequences = []

    for user_id, history in train_histories.items():

        for target_idx in range(1, len(history)):

            start_idx = max(0, target_idx - max_seq_len)

            input_sequence = history[start_idx:target_idx]
            target_item = history[target_idx]

            sequences.append(
                (
                    user_id,
                    input_sequence,
                    target_item,
                )
            )

    return sequences