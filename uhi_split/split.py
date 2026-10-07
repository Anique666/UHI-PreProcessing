"""Reproducible spatial train/val/test partition of the BBMP modeling area.

The partition is a directional-quantile partition, NOT an official
administrative boundary. It is explicitly artificial, contiguous by
construction, reproducible, configurable and saved to disk.

Split semantics (configurable fractions):
    test  = north-most test_fraction of the modeling area (+ auto buffer)
    val   = east-most val_fraction of the remaining area
    train = everything else (central/south/west)

A positive assignment_buffer_m expands the train region into the boundary zone
between test and the rest (never invents pixels).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask

from .config import SplitConfig
from .months import GridInfo, grid_to_affine

# Split ids used consistently across the pipeline.
TRAIN, VAL, TEST, OUTSIDE = 1, 2, 3, 0
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


@dataclass
class SpatialSplit:
    region_mask: np.ndarray      # (H, W) bool  - within BBMP modeling area
    split_mask: np.ndarray       # (H, W) uint8 - TRAIN/VAL/TEST/OUTSIDE
    transform: tuple[float, ...]
    crs: str
    stats: dict

    def save(self, out_dir: Path) -> None:
        from affine import Affine

        out_dir.mkdir(parents=True, exist_ok=True)
        prof = dict(
            driver="GTiff",
            height=self.region_mask.shape[0],
            width=self.region_mask.shape[1],
            count=1,
            dtype="uint8",
            crs=self.crs,
            transform=Affine.from_gdal(*self.transform),
            compress="lzw",
        )
        with rasterio.open(out_dir / "split_mask.tif", "w", **prof) as ds:
            ds.write(self.split_mask.astype(np.uint8), 1)
        with rasterio.open(out_dir / "region_mask.tif", "w", **prof) as ds:
            ds.write(self.region_mask.astype(np.uint8), 1)


def load_bbmp_union(geojson: Path, target_crs) -> object:
    import geopandas as gpd
    from shapely import make_valid

    g = gpd.read_file(geojson)
    if g.empty:
        raise ValueError(f"BBMP boundary is empty: {geojson}")
    n_invalid = int((~g.geometry.is_valid).sum())
    g["geometry"] = g.geometry.apply(make_valid)
    if g.geometry.is_valid.sum() != len(g):
        raise ValueError("BBMP geometries still invalid after make_valid")
    gp = g.to_crs(target_crs)
    return gp.geometry.union_all(), len(g), n_invalid


def create_spatial_split(
    grid: GridInfo, boundary: Path, cfg: SplitConfig
) -> SpatialSplit:
    transform = grid_to_affine(grid)
    W, H = grid.width, grid.height

    union, n_wards, _ = load_bbmp_union(boundary, grid.crs)
    region = geometry_mask(
        [union], out_shape=(H, W), transform=transform, invert=True
    )
    if region.sum() == 0:
        raise ValueError("BBMP footprint does not intersect the raster grid")

    rows, cols = np.nonzero(region)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e

    theta = np.deg2rad(cfg.rotation_deg)
    # Rotated coordinates: north-up at rotation 0.
    Yr = ys * np.cos(theta) + xs * np.sin(theta)
    Xr = xs * np.cos(theta) - ys * np.sin(theta)

    # TEST = north-most fraction.
    test_cut = np.quantile(Yr, 1.0 - cfg.test_fraction)
    test_sel = Yr >= test_cut
    rem = ~test_sel

    # Optional train-only buffer: pull the test boundary further north so the
    # strata are separated by a train-only gap. Never leaves a gap.
    if cfg.assignment_buffer_m and cfg.assignment_buffer_m > 0:
        buf_sel = (Yr < test_cut) & (Yr >= test_cut - cfg.assignment_buffer_m)
        test_sel = test_sel & ~buf_sel
        rem = ~test_sel

    # VAL = east-most fraction of the remaining area.
    val_cut = np.quantile(Xr[rem], 1.0 - cfg.val_fraction / (1.0 - cfg.test_fraction))
    val_sel = rem & (Xr >= val_cut)
    train_sel = ~(test_sel | val_sel)

    split = np.zeros((H, W), dtype=np.uint8)
    split[rows[train_sel], cols[train_sel]] = TRAIN
    split[rows[val_sel], cols[val_sel]] = VAL
    split[rows[test_sel], cols[test_sel]] = TEST

    n = int(region.sum())
    stats = {
        "split_method": cfg.method,
        "rotation_deg": cfg.rotation_deg,
        "test_fraction": cfg.test_fraction,
        "val_fraction": cfg.val_fraction,
        "assignment_buffer_m": cfg.assignment_buffer_m,
        "n_wards_in_boundary": n_wards,
        "boundary_source": str(boundary),
        "is_official_admin_boundary": False,
        "note": (
            "Artificial directional-quantile partition. 'Central+South'/'East'/"
            "'North+buffer' are NOT official administrative names; they are "
            "directional descriptions of this partition. Human approval required "
            "before full run."
        ),
        "region_pixels": n,
        "pixels": {
            name: int((split == sid).sum()) for sid, name in SPLIT_NAMES.items()
        },
        "pixel_pct": {
            name: 100.0 * int((split == sid).sum()) / n for sid, name in SPLIT_NAMES.items()
        },
        "region_area_km2": float(n * 30.0 * 30.0 / 1e6),
    }

    for sid, name in SPLIT_NAMES.items():
        if (split == sid).sum() < cfg.min_pixels_per_split:
            raise ValueError(
                f"Split '{name}' has fewer than min_pixels_per_split="
                f"{cfg.min_pixels_per_split}; refusing to continue."
            )

    return SpatialSplit(
        region_mask=region,
        split_mask=split,
        transform=grid.transform,
        crs=grid.crs,
        stats=stats,
    )