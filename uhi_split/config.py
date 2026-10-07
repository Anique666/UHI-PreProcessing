"""Configuration loading and validation for the UHI tensor-building stage."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Paths:
    input_dir: Path
    output_dir: Path
    bbmp_boundary: Path


@dataclass
class DataConfig:
    expected_start: str
    expected_end: str
    sequence_length: int
    patch_size: int
    stride: int
    purity_threshold: float
    channel_names: list[str]
    lst_channel: str
    terrain_channels: list[str]
    max_gap_fraction: float = 0.10
    max_open_rasters: int = 24


@dataclass
class SplitConfig:
    method: str
    rotation_deg: float
    test_fraction: float
    val_fraction: float
    assignment_buffer_m: float
    min_pixels_per_split: int


@dataclass
class Config:
    paths: Paths
    data: DataConfig
    split: SplitConfig
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in d["paths"].items()}
        return d


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key '{key}' in {where}")
    return d[key]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)

    paths = Paths(
        input_dir=Path(_require(raw, "paths", "root")["input_dir"]),
        output_dir=Path(raw["paths"]["output_dir"]),
        bbmp_boundary=Path(raw["paths"]["bbmp_boundary"]),
    )
    data = DataConfig(**raw["data"])
    split = SplitConfig(**raw["split"])

    _validate(paths, data, split)
    return Config(paths=paths, data=data, split=split, raw=raw)


def _validate(paths: Paths, data: DataConfig, split: SplitConfig) -> None:
    if not paths.input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {paths.input_dir}")
    if not paths.bbmp_boundary.exists():
        raise FileNotFoundError(f"bbmp_boundary does not exist: {paths.bbmp_boundary}")
    if data.sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if data.patch_size <= 0 or data.stride <= 0:
        raise ValueError("patch_size and stride must be > 0")
    if data.max_open_rasters < 1:
        raise ValueError("max_open_rasters must be >= 1")
    if not (0.0 < data.purity_threshold <= 1.0):
        raise ValueError("purity_threshold must be in (0, 1]")
    if split.method != "directional_quantile":
        raise ValueError(f"Unsupported split method: {split.method}")
    if not (0.0 < split.test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1)")
    if not (0.0 < split.val_fraction < 1.0 - split.test_fraction):
        raise ValueError("val_fraction must leave room for train")
    if len(set(data.channel_names)) != len(data.channel_names):
        raise ValueError("channel_names contains duplicates")
    if data.lst_channel not in data.channel_names:
        raise ValueError("lst_channel must be one of channel_names")
    for c in data.terrain_channels:
        if c not in data.channel_names:
            raise ValueError(f"terrain channel {c} not in channel_names")


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)