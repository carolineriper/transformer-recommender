import torch
from torch.utils.data import Dataset


class MovieSequenceDataset(Dataset):
    """Pad next-item examples and retain user IDs for evaluation masking."""

    def __init__(
        self,
        sequences: list[tuple[int, list[int], int]],
        max_seq_len: int,
        padding_idx: int,
    ) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        user_id, sequence, target = self.sequences[idx]

        sequence = sequence[-self.max_seq_len:]
        padding_length = self.max_seq_len - len(sequence)

        sequence = [self.padding_idx] * padding_length + sequence

        x = torch.tensor(sequence, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)

        return x, y, user_id
