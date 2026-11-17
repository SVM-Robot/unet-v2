"""Per-chip ERA5 wind keyed by manifest sample_id."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class WindTable:
    """Nearest-cell u10/v10, normalized with the fixed train-split constants."""

    def __init__(self, path: str | Path):
        wind = np.load(path)
        required = {"sample_id", "split", "u10", "v10"}
        missing = required - set(wind.files)
        if missing:
            raise ValueError(f"Wind file {path} is missing arrays: {sorted(missing)}")
        sample_id = wind["sample_id"].astype(str)
        if len(sample_id) != len(set(sample_id)):
            raise ValueError(f"Wind file {path} has duplicate sample_id values")
        self.split = wind["split"].astype(str)
        self.u10 = np.asarray(wind["u10"], dtype=np.float32)
        self.v10 = np.asarray(wind["v10"], dtype=np.float32)
        train = self.split == "train"
        if not train.any():
            raise ValueError(f"Wind file {path} has no train rows for normalization")
        # Z-score, fixed from the train split: (u - mean) / std.
        self.mean = np.array([-0.48713719844818115, -0.008137172088027], dtype=np.float32)
        self.std = np.array([3.263791561126709, 2.9456822872161865], dtype=np.float32)
        # RMS speed scale: s = sqrt(mean(u**2 + v**2)) over the train split, then u/s and v/s.
        # s = float(np.sqrt(np.mean(self.u10[train].astype(np.float64) ** 2 + self.v10[train].astype(np.float64) ** 2)))
        # self.mean = np.zeros(2, dtype=np.float32)
        # self.std = np.full(2, s, dtype=np.float32)
        self._index = {sample: i for i, sample in enumerate(sample_id)}

    def uv(self, sample_id: str) -> np.ndarray:
        try:
            index = self._index[sample_id]
        except KeyError:
            raise KeyError(f"No wind row for sample_id {sample_id!r}") from None
        raw = np.array([self.u10[index], self.v10[index]], dtype=np.float32)
        return (raw - self.mean) / (self.std + np.float32(1e-6))
