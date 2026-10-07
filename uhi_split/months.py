"""Discovery, chronological indexing and structural validation of monthly TIFFs."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

MONTH_RE = re.compile(r"(?P<year>\d{4})_(?P<month>\d{2})")


@dataclass
class MonthRecord:
    index: int
    year: int
    month: int
    path: Path

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def key(self) -> tuple[int, int]:
        return (self.year, self.month)


def _iter_months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def discover_months(input_dir: Path) -> list[MonthRecord]:
    """Find monthly 18-feature TIFFs, order them chronologically, require
    completeness over the expected range, and fail loudly otherwise."""
    tifs = sorted(p for p in input_dir.rglob("*.tif") if "18features" in p.name)
    if not tifs:
        raise FileNotFoundError(f"No *18features.tif found under {input_dir}")

    found: dict[tuple[int, int], Path] = {}
    for p in tifs:
        m = MONTH_RE.search(p.stem)
        if not m:
            raise ValueError(f"Cannot parse YYYY_MM from {p.name}")
        key = (int(m.group("year")), int(m.group("month")))
        if key in found:
            raise ValueError(f"Duplicate month {key}: {found[key]} and {p}")
        found[key] = p

    keys = sorted(found)
    expected = _iter_months(keys[0], keys[-1])
    missing = [k for k in expected if k not in found]
    if missing:
        raise ValueError(
            "Missing months in the discovered range: "
            + ", ".join(f"{y:04d}-{m:02d}" for y, m in missing)
        )

    return [MonthRecord(i, y, m, found[(y, m)]) for i, (y, m) in enumerate(keys)]


@dataclass
class GridInfo:
    width: int
    height: int
    count: int
    crs: str
    transform: tuple[float, ...]
    dtype: str
    descriptions: tuple[str | None, ...]


def inspect_grid(months: list[MonthRecord]) -> GridInfo:
    """Verify every month shares CRS/transform/dimensions/band ordering/dtype and
    that band descriptions match the expected channel order. Fail loudly on any
    disagreement."""
    ref: rasterio.DatasetReader | None = None
    ref_info: GridInfo | None = None
    try:
        for rec in months:
            with rasterio.open(rec.path) as ds:
                info = GridInfo(
                    width=ds.width,
                    height=ds.height,
                    count=ds.count,
                    crs="" if ds.crs is None else ds.crs.to_string(),
                    transform=tuple(round(v, 10) for v in ds.transform.to_gdal()),
                    dtype=ds.dtypes[0],
                    descriptions=tuple(ds.descriptions),
                )
                if ref_info is None:
                    ref_info = info
                elif info != ref_info:
                    raise ValueError(
                        f"Grid/metadata mismatch at {rec.path.name}.\n"
                        f"  expected: {ref_info}\n  found:    {info}"
                    )
    finally:
        if ref is not None:  # pragma: no cover - defensive
            ref.close()
    assert ref_info is not None
    return ref_info


def check_channel_order(grid: GridInfo, expected: list[str]) -> None:
    descs = list(grid.descriptions)
    if descs == expected:
        return
    if all(d is None for d in descs):
        raise ValueError(
            "TIFF has no band descriptions; cannot confirm the 18-channel order "
            "that the science requires. Refusing to guess."
        )
    raise ValueError(
        "Band description order does not match the configured channel order.\n"
        f"  TIFF:     {descs}\n  expected: {expected}"
    )


def grid_to_affine(grid: GridInfo):
    from affine import Affine

    t = grid.transform
    # to_gdal() order: c, a, b, f, d, e  -> Affine(a,b,c,d,e,f)
    return Affine(t[1], t[2], t[0], t[4], t[5], t[3])