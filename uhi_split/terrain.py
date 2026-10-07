"""Precompute the per-pixel temporal mean of static terrain channels.

The validated preprocessing replaces each terrain channel (Elevation, Slope)
with its per-pixel temporal mean over the run's months, applied identically to
every timestep (the previous materialized-tensor implementation did this via a
per-run nanmean; the lazy Dataset reproduces it exactly).

For the lazy architecture we must reproduce that EXACTLY without reading all
72 monthly TIFFs for every sample. We therefore precompute the per-pixel
temporal mean once and store it as a tiny 2-band raster. The Dataset then reads
a 128x128 window of this raster instead of 72 monthly terrain windows.

This does not change the terrain treatment; it is an exact, lossless
precomputation of it (nanmean over the configured months).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from .months import GridInfo, MonthRecord
from .config import DataConfig


def compute_terrain_mean(
    months: list[MonthRecord], grid: GridInfo, data: DataConfig
) -> np.ndarray:
    """Per-pixel nanmean over months for each terrain channel -> [K, H, W]."""
    terrain_names = list(data.terrain_channels)
    if not terrain_names:
        return np.zeros((0, grid.height, grid.width), dtype=np.float32)
    idx = [data.channel_names.index(c) for c in terrain_names]
    H, W = grid.height, grid.width
    total = np.zeros((len(idx), H, W), dtype=np.float64)
    count = np.zeros((len(idx), H, W), dtype=np.float64)

    for rec in months:
        with rasterio.open(rec.path) as ds:
            for k, ch in enumerate(idx):
                band = ds.read(ch + 1).astype(np.float64)
                finite = np.isfinite(band)
                total[k][finite] += band[finite]
                count[k][finite] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1.0), np.nan)
    return mean.astype(np.float32)


def save_terrain_mean(
    arr: np.ndarray, grid: GridInfo, data: DataConfig, out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    k = arr.shape[0]
    prof = dict(
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=max(k, 1),
        dtype="float32",
        crs=grid.crs,
        transform=Affine.from_gdal(*grid.transform),
        nodata=np.nan,
        compress="lzw",
    )
    if k == 0:
        raise ValueError("No terrain channels configured; nothing to save")
    with rasterio.open(out_path, "w", **prof) as ds:
        ds.write(arr.astype(np.float32))
        for i, name in enumerate(data.terrain_channels):
            ds.set_band_description(i + 1, name)


def load_terrain_mean(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
        descs = tuple(ds.descriptions)
    return arr, descs