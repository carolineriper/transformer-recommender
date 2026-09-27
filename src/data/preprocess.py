import pandas as pd


def sort_interactions(ratings: pd.DataFrame) -> pd.DataFrame:
    """Sort interactions chronologically within each user."""
    return ratings.sort_values(
        by=["user_id", "timestamp"],
        ignore_index=True,
    )


def build_user_histories(
    ratings: pd.DataFrame,
) -> dict[int, list[int]]:
    """Map each user to an ordered list of dense ``movie_idx`` values."""
    return (
        ratings
        .groupby("user_id", sort=False)["movie_idx"]
        .apply(list)
        .to_dict()
    )


def temporal_split(
    user_histories: dict[int, list[int]],
) -> tuple[
    dict[int, list[int]],
    dict[int, int],
    dict[int, int],
]:
    """Use each user's final two interactions as validation and test targets."""
    train_histories: dict[int, list[int]] = {}
    val_targets: dict[int, int] = {}
    test_targets: dict[int, int] = {}

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
    """Generate sliding-window next-item examples from training histories."""
    sequences: list[tuple[int, list[int], int]] = []

    for user_id, history in train_histories.items():
        for target_idx in range(1, len(history)):
            start_idx = max(0, target_idx - max_seq_len)
            input_sequence = history[start_idx:target_idx]
            target_item = history[target_idx]
            sequences.append((user_id, input_sequence, target_item))

    return sequences


def generate_validation_sequences(
    train_histories: dict[int, list[int]],
    val_targets: dict[int, int],
) -> list[tuple[int, list[int], int]]:
    """Create one validation next-item example per user."""
    return [
        (user_id, history, val_targets[user_id])
        for user_id, history in train_histories.items()
    ]


def generate_test_sequences(
    train_histories: dict[int, list[int]],
    val_targets: dict[int, int],
    test_targets: dict[int, int],
) -> list[tuple[int, list[int], int]]:
    """Create test examples with the validation item added to known history."""
    return [
        (
            user_id,
            history + [val_targets[user_id]],
            test_targets[user_id],
        )
        for user_id, history in train_histories.items()
    ]
