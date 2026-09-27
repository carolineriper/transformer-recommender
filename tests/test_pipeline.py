import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import MovieSequenceDataset
from src.data.preprocess import (
    generate_sequences,
    generate_test_sequences,
    generate_validation_sequences,
    temporal_split,
)
from src.evaluation.metrics import evaluate_model
from src.models.transformer_recommender import TransformerRecommender


class FixedScoreModel(nn.Module):
    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))

    def forward(self, movie_ids: torch.Tensor) -> torch.Tensor:
        return self.scores.unsqueeze(0).expand(movie_ids.size(0), -1).clone()


class PipelineTests(unittest.TestCase):
    def test_temporal_split_and_examples(self) -> None:
        histories = {1: [0, 1, 2, 3, 4]}

        train, validation, test = temporal_split(histories)

        self.assertEqual(train, {1: [0, 1, 2]})
        self.assertEqual(validation, {1: 3})
        self.assertEqual(test, {1: 4})
        self.assertEqual(
            generate_sequences(train, max_seq_len=2),
            [(1, [0], 1), (1, [0, 1], 2)],
        )
        self.assertEqual(
            generate_validation_sequences(train, validation),
            [(1, [0, 1, 2], 3)],
        )
        self.assertEqual(
            generate_test_sequences(train, validation, test),
            [(1, [0, 1, 2, 3], 4)],
        )

    def test_dataset_returns_user_id_and_left_padding(self) -> None:
        dataset = MovieSequenceDataset(
            sequences=[(7, [1, 2], 3)],
            max_seq_len=4,
            padding_idx=5,
        )

        x, y, user_id = dataset[0]

        self.assertEqual(x.tolist(), [5, 5, 1, 2])
        self.assertEqual(y.item(), 3)
        self.assertEqual(user_id, 7)

    def test_full_history_masking_changes_top_result(self) -> None:
        dataset = MovieSequenceDataset(
            sequences=[(1, [0, 1], 2)],
            max_seq_len=3,
            padding_idx=4,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = FixedScoreModel([0.0, 10.0, 5.0, 1.0])

        hit_rate, ndcg = evaluate_model(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            seen_histories={1: [0, 1]},
            k=1,
        )

        self.assertEqual(hit_rate, 1.0)
        self.assertEqual(ndcg, 1.0)

    def test_transformer_output_shape(self) -> None:
        model = TransformerRecommender(
            num_movies=5,
            embedding_dim=8,
            max_seq_len=4,
            num_heads=2,
            hidden_dim=16,
            num_layers=2,
        )
        movie_ids = torch.tensor([[5, 0, 1, 2]], dtype=torch.long)

        with torch.inference_mode():
            logits = model(movie_ids)

        self.assertEqual(tuple(logits.shape), (1, 5))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
