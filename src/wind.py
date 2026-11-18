"""Per-chip ERA5 wind keyed by manifest sample_id."""
from __future__ import annotations

from pathlib import Path

import numpy as np

WIND_MODES = (
    "none",
    "point_channels",
    "film_uv",
    "film_uvw",
    "film_grid_flat",
    "film_grid_cnn",
)
FILM_GRID_MODES = ("film_grid_flat", "film_grid_cnn")
WIND_NORMS = ("normal", "scale")
CNN_WIND_DIM = 64

# Fixed train-split z-score of nearest-cell u10 and v10.
_U_MEAN = np.float32(-0.48713719844818115)
_U_STD = np.float32(3.263791561126709)
_V_MEAN = np.float32(-0.008137172088027)
_V_STD = np.float32(2.9456822872161865)


def resolve_wind_mode(mode: str, point_npz: str | None, grid_npz: str | None) -> str:
    """`--wind-point-npz` alone keeps the existing point-channel runs working."""
    if mode not in WIND_MODES:
        raise ValueError(f"Unsupported wind_mode {mode!r}")
    if mode == "none" and point_npz and not grid_npz:
        return "point_channels"
    return mode


def open_wind_table(
    mode: str,
    point_npz: str | None,
    grid_npz: str | None,
    norm: str = "normal",
) -> "WindTable | None":
    if mode == "none":
        return None
    if mode in ("point_channels", "film_uv"):
        if not point_npz:
            raise ValueError(f"{mode} requires --wind-point-npz")
        return WindTable(point_npz, features="uv", norm=norm)
    if mode == "film_uvw":
        if not point_npz:
            raise ValueError("film_uvw requires --wind-point-npz")
        return WindTable(point_npz, features="uvw", norm=norm)
    if mode in FILM_GRID_MODES:
        if not grid_npz or not point_npz:
            raise ValueError(f"{mode} requires --wind-grid-npz and --wind-point-npz")
        return WindTable(grid_npz, features="grid", split_npz=point_npz, norm=norm)
    raise ValueError(f"Unsupported wind_mode {mode!r}")


def film_wind_dim(mode: str, table: "WindTable | None") -> int:
    if mode == "film_uv":
        return 2
    if mode == "film_uvw":
        return 3
    if mode == "film_grid_flat":
        return table.channels * table.grid_size * table.grid_size
    if mode == "film_grid_cnn":
        return CNN_WIND_DIM
    return 0


class WindTable:
    """Normalized wind for one sample. Point values use fixed constants; grids use the train split."""

    def __init__(
        self,
        path: str | Path,
        features: str = "uv",
        split_npz: str | Path | None = None,
        norm: str = "normal",
    ):
        if features not in {"uv", "uvw", "grid"}:
            raise ValueError(f"Unsupported wind features {features!r}")
        if norm not in WIND_NORMS:
            raise ValueError(f"Unsupported wind norm {norm!r}. Choose from {WIND_NORMS}")
        self.features = features
        self.norm = norm
        self.grid_size = 0
        self.channels = 2
        wind = np.load(path, allow_pickle=features == "grid")
        sample_id = wind["sample_id"].astype(str)
        if len(sample_id) != len(set(sample_id)):
            raise ValueError(f"Wind file {path} has duplicate sample_id values")
        self._index = {sample: i for i, sample in enumerate(sample_id)}

        if features in {"uv", "uvw"}:
            missing = {"u10", "v10"} - set(wind.files)
            if missing:
                raise ValueError(f"Wind file {path} is missing arrays: {sorted(missing)}")
            u = np.asarray(wind["u10"], dtype=np.float32)
            v = np.asarray(wind["v10"], dtype=np.float32)
            if norm == "scale" or features == "uvw":
                if "split" not in wind.files:
                    raise ValueError(f"Wind file {path} needs split to normalize wind")
                train = wind["split"].astype(str) == "train"
                if not train.any():
                    raise ValueError(f"Wind file {path} has no train rows")
            if features == "uv":
                raw = np.stack([u, v], axis=1)
                if norm == "normal":
                    self.mean = np.array([_U_MEAN, _V_MEAN], dtype=np.float32)
                    self.std = np.array([_U_STD, _V_STD], dtype=np.float32)
                else:
                    scale = _rms_scale(u[train], v[train])
                    self.mean = np.zeros(2, dtype=np.float32)
                    self.std = np.full(2, scale, dtype=np.float32)
            else:
                speed = np.hypot(u, v)
                raw = np.stack([u, v, speed], axis=1)
                self.channels = 3
                if norm == "normal":
                    self.mean = np.array([_U_MEAN, _V_MEAN, speed[train].mean()], dtype=np.float32)
                    self.std = np.array([_U_STD, _V_STD, speed[train].std()], dtype=np.float32)
                else:
                    scale = _rms_scale(u[train], v[train])
                    self.mean = np.zeros(3, dtype=np.float32)
                    self.std = np.full(3, scale, dtype=np.float32)
            self._values = (raw - self.mean) / (self.std + np.float32(1e-6))
            return

        missing = {"u", "v", "sample_id"} - set(wind.files)
        if missing:
            raise ValueError(f"Wind file {path} is missing arrays: {sorted(missing)}")
        if split_npz is None:
            raise ValueError("Grid wind requires the point npz so the train split is known")
        grids = np.stack([wind["u"], wind["v"]], axis=1).astype(np.float32)
        if np.isnan(grids).any():
            raise ValueError(f"Wind file {path} contains NaN values")
        split = np.load(split_npz)
        split_by_id = dict(zip(split["sample_id"].astype(str), split["split"].astype(str)))
        try:
            train = np.array([split_by_id[sample] == "train" for sample in sample_id])
        except KeyError as exc:
            raise KeyError(f"Grid sample_id {exc} is missing from {split_npz}") from None
        if not train.any():
            raise ValueError(f"No train rows found for grid file {path}")
        self.grid_size = int(grids.shape[-1])
        if grids.shape[-2] != self.grid_size:
            raise ValueError(f"Expected a square wind grid, got {grids.shape}")
        train_grids = grids[train]
        if norm == "normal":
            self.mean = train_grids.mean(axis=(0, 2, 3)).astype(np.float32)
            self.std = train_grids.std(axis=(0, 2, 3)).astype(np.float32)
        else:
            scale = _rms_scale(train_grids[:, 0], train_grids[:, 1])
            self.mean = np.zeros(2, dtype=np.float32)
            self.std = np.full(2, scale, dtype=np.float32)
        self._values = (grids - self.mean.reshape(1, 2, 1, 1)) / (self.std.reshape(1, 2, 1, 1) + np.float32(1e-6))

    def condition(self, sample_id: str) -> np.ndarray:
        try:
            index = self._index[sample_id]
        except KeyError:
            raise KeyError(f"No wind row for sample_id {sample_id!r}") from None
        return self._values[index].astype(np.float32, copy=False)


def _rms_scale(u: np.ndarray, v: np.ndarray) -> np.float32:
    """s = sqrt(mean(u^2 + v^2)) over the given train values."""
    return np.float32(np.sqrt(np.mean(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)))
