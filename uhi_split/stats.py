"""Train-only fill and normalization statistics.

All statistics are computed from TRAINING patches only, restricted to pixels
whose split id is TRAIN. Validation/test data never influence these values.

Terrain channels (Elevation, Slope) are static across months. For those, a
per-patch temporal mean is computed first, matching exactly how the terrain
value is supplied to the model at normalize time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from .config import DataConfig
from .months import MonthRecord
from .patches import PatchIndex
from .split import TRAIN

_MIN_STD = 1e-8


@dataclass
class FillNormalization:
    channel_names: list[str]
    fill_values: list[float]
    means: list[float]
    stds: list[float]
    counts: list[int]
    gap_fraction: list[float]
    degenerate_channels: list[str] = field(default_factory=list)
    over_gap_channels: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "channel_names": self.channel_names,
            "fill_values": self.fill_values,
            "train_means": self.means,
            "train_stds": self.stds,
            "valid_pixel_counts": self.counts,
            "gap_fraction": self.gap_fraction,
            "degenerate_channels": self.degenerate_channels,
            "over_gap_channels": self.over_gap_channels,
            "min_std_guard": _MIN_STD,
        }


def _read_window(path: Path, r0: int, c0: int, size: int) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(window=((r0, r0 + size), (c0, c0 + size))).astype(np.float64)


def compute_train_stats(
    months: list[MonthRecord],
    train_patches: list[PatchIndex],
    split_mask: np.ndarray,
    data: DataConfig,
    terrain_mean: np.ndarray,
    max_gap_fraction: float = 0.10,
) -> FillNormalization:
    """TRAIN-only fill/normalization statistics.

    Dynamic channels are read from the monthly TIFFs; terrain channels use the
    precomputed per-pixel temporal-mean raster (exactly equal to the nanmean
    over `months` that the validated preprocessing applies to terrain).
    """
    n_ch = len(data.channel_names)
    terrain_idx = [data.channel_names.index(c) for c in data.terrain_channels]
    if not train_patches:
        raise ValueError("No training patches available to compute statistics")
    if len(terrain_idx) != terrain_mean.shape[0]:
        raise ValueError(
            f"terrain_mean has {terrain_mean.shape[0]} bands but "
            f"{len(terrain_idx)} terrain channels are configured"
        )

    acc = [{"n": 0, "s": 0.0, "ss": 0.0, "nmax": 0} for _ in range(n_ch)]

    for pi in train_patches:
        r0, c0, size = pi.row0, pi.col0, pi.size
        train_px = split_mask[r0 : r0 + size, c0 : c0 + size] == TRAIN
        n_px = int(train_px.sum())

        for rec in months:
            cube = _read_window(rec.path, r0, c0, size)  # [C, s, s]
            for ch in range(n_ch):
                if ch in terrain_idx:
                    continue
                acc[ch]["nmax"] += n_px
                vals = cube[ch][train_px]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    acc[ch]["n"] += int(vals.size)
                    acc[ch]["s"] += float(vals.sum())
                    acc[ch]["ss"] += float((vals * vals).sum())

        for j, ch in enumerate(terrain_idx):
            tmean = terrain_mean[j, r0 : r0 + size, c0 : c0 + size]
            acc[ch]["nmax"] += n_px
            vals = tmean[train_px]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                acc[ch]["n"] += int(vals.size)
                acc[ch]["s"] += float(vals.sum())
                acc[ch]["ss"] += float((vals * vals).sum())

    means, stds, fills, counts, gaps, degenerate = [], [], [], [], [], []
    for ch, name in enumerate(data.channel_names):
        a = acc[ch]
        if a["n"] == 0:
            raise ValueError(
                f"No valid training pixels for channel {name}; cannot compute "
                "statistics. Refusing to fabricate a value."
            )
        mean = a["s"] / a["n"]
        var = a["ss"] / a["n"] - mean ** 2
        std = float(np.sqrt(max(var, 0.0)))
        if std <= _MIN_STD:
            degenerate.append(name)
            std_used = 1.0
        else:
            std_used = std
        means.append(float(mean))
        stds.append(float(std_used))
        fills.append(float(mean))
        counts.append(int(a["n"]))
        gaps.append(float(1.0 - a["n"] / a["nmax"]) if a["nmax"] else np.nan)

    over = [n for n, g in zip(data.channel_names, gaps) if g > max_gap_fraction]
    return FillNormalization(
        channel_names=list(data.channel_names),
        fill_values=fills,
        means=means,
        stds=stds,
        counts=counts,
        gap_fraction=gaps,
        degenerate_channels=degenerate,
        over_gap_channels=over,
    )