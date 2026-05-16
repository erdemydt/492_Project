"""Model zoo.

Every model exposes the same interface: fit(X, y) and predict(X), where X
is a 2-D float array of scaled features (rows in chronological order) and y
is the next-day return. The LSTM rebuilds short sequences internally from
the row order, so it sees the same information as the tabular models.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


class NaivePersistence:
    """Predict tomorrow's return = today's return (column 0 == ret_1).

    A deliberately trivial baseline. If sophisticated models cannot beat
    this, they are not adding value.
    """

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


class TorchLSTM:
    """Single-layer LSTM regressor (PyTorch, CPU).

    Builds length-`seq_len` sequences from consecutive feature rows. A
    sequence ending at row i only ever uses rows <= i, so using train-era
    context for an early test prediction is past data, not leakage.
    """

    def __init__(self, seq_len=20, hidden=32, epochs=40, lr=1e-3, seed=0):
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self._fitted = False

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def _make_seqs(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        n, f = X.shape
        idx = range(self.seq_len - 1, n)
        seqs = np.stack([X[i - self.seq_len + 1 : i + 1] for i in idx])
        if y is None:
            return seqs
        return seqs, np.asarray(y, dtype=np.float32)[self.seq_len - 1 :]

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        seqs, tgt = self._make_seqs(X, y)
        self._n_features = seqs.shape[2]

        class Net(nn.Module):
            def __init__(self, f, h):
                super().__init__()
                self.lstm = nn.LSTM(f, h, batch_first=True)
                self.head = nn.Linear(h, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = Net(self._n_features, self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        xb = torch.from_numpy(seqs)
        yb = torch.from_numpy(tgt)

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xb), yb)
            loss.backward()
            opt.step()
        self._fitted = True
        return self

    def predict(self, X):
        import torch

        X = np.asarray(X, dtype=np.float32)
        seqs = self._make_seqs(X)
        self.net.eval()
        with torch.no_grad():
            pred = self.net(torch.from_numpy(seqs)).numpy()
        # First seq_len-1 rows lack a full window: fall back to flat (0.0).
        pad = np.zeros(self.seq_len - 1, dtype=np.float32)
        return np.concatenate([pad, pred])


def build_models(seq_len: int = 20):
    """Return an ordered dict of name -> fresh model instance."""
    models = {
        "Naive(persistence)": NaivePersistence(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=0, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=2, learning_rate=0.03, subsample=0.8, random_state=0
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=0,
        ),
    }
    if TorchLSTM.available():
        models["LSTM(torch)"] = TorchLSTM(seq_len=seq_len)
    return models
