# Transformer Recommender

A from-scratch sequential recommender built with PyTorch and evaluated on
[MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). The project uses
a causal Transformer to predict the next movie in a user's interaction history
and compares it with a non-personalized popularity baseline.

The implementation is intentionally compact and educational: movie and
positional embeddings, multi-head causal self-attention, feed-forward blocks,
training, checkpoint selection, ranking metrics, and seen-item filtering are
implemented explicitly rather than hidden behind a recommender framework.

## Results

Final metrics are reported on the held-out temporal test item for each of the
6,040 users. Both methods exclude every item already known at test time.

| Model | HR@10 | NDCG@10 |
|---|---:|---:|
| **Transformer + full-history masking** | **0.2954** | **0.1714** |
| Popularity baseline | 0.0369 | 0.0180 |

The Transformer places the actual next movie in its Top-10 for 29.54% of users,
compared with 3.69% for the popularity baseline. NDCG additionally rewards
placing the target near the top of the recommendation list.

## Problem formulation

Ratings are treated as timestamped interaction events. The model receives up to
the last 50 movie indices and predicts one next item over the 3,706 movies that
have observed ratings.

For each user, interactions are sorted by timestamp and split chronologically:

```text
earlier interactions          penultimate item       final item
        train            ->      validation      ->     test
```

- Train examples are generated with a sliding next-item prediction window.
- Validation uses the full train history to predict the penultimate item.
- Test uses the train history plus the validation item to predict the final item.

This protocol prevents future interactions from leaking into model training or
checkpoint selection.

## Model architecture

```text
movie indices
    -> learned movie embeddings + learned positional embeddings
    -> 2 pre-norm Transformer blocks
         - 4-head causal self-attention
         - padding-key masking
         - residual connections
         - GELU feed-forward network
    -> final LayerNorm
    -> representation at the last sequence position
    -> linear projection over all candidate movies
```

Training configuration:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 50 |
| Embedding size | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward hidden size | 512 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Objective | Cross-entropy over all movies |
| Completed training | 6 epochs |

The best checkpoint is selected by validation NDCG@10. The test split is used
only once for the final report and is not used to select an epoch or tune the
model.

## Evaluation protocol

The model context and candidate filtering serve different purposes:

- the Transformer context is limited to the last 50 interactions;
- candidate filtering uses the user's complete known history.

`MovieSequenceDataset` therefore returns `(x, y, user_id)`. During evaluation,
`user_id` is used to mask all previously seen movie indices before `topk` is
computed. At validation time the mask is the train history. At test time it is
the train history plus the validation item. The target itself is never added to
the mask.

The popularity baseline counts movies only in the training histories, ranks them
globally, and applies the same full-history filtering before producing Top-10
recommendations. This keeps the candidate protocol comparable across models.

## Repository layout

```text
transformer-recommender/
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_PyTorch_Basics.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── models/
│       ├── embedding.py
│       ├── feed_forward.py
│       ├── multi_head_attention.py
│       ├── positional_embedding.py
│       ├── transformer_block.py
│       └── transformer_recommender.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Reproducing the project

The verified environment uses Python 3.12.2. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download MovieLens 1M from the official GroupLens page and extract it so the
following files exist:

```text
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

Then start Jupyter from the repository root:

```bash
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_PyTorch_Basics.ipynb`

The second notebook is compatible with **Restart Kernel -> Run All**. If
`checkpoints/best_model.pt` exists, it loads the checkpoint and skips training.
If it is missing, the notebook trains the configured six epochs and saves the
best validation checkpoint. Local checkpoints and raw data are intentionally
excluded from Git.

Run the fast, training-free checks with:

```bash
python -m unittest discover -s tests -v
```

## Notes and scope

- Movie genres and rating values are not model features; the first version uses
  interaction order only.
- The model uses a full softmax over the observed movie vocabulary.
- This repository deliberately stops at the completed first model version: no
  hyperparameter search, sampled loss, side features, or serving layer is added.
- The MovieLens dataset is not redistributed. Review the terms included with the
  dataset before use.
