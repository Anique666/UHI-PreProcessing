"""Standalone UHI spatial-split / normalization / lazy-dataset package. 

The authoritative PyTorch Dataset lives in ``uhi_split/dataset.py`` and is
copied verbatim into the processed-dataset directory as ``dataset.py``.
"""
from .config import Config, load_config
from .months import discover_months, inspect_grid, check_channel_order
from .split import create_spatial_split, SpatialSplit, TRAIN, VAL, TEST, OUTSIDE
from .patches import generate_patch_index, PatchIndex, patch_starts
from .terrain import compute_terrain_mean, save_terrain_mean, load_terrain_mean

__all__ = [
    "Config",
    "load_config",
    "discover_months",
    "inspect_grid",
    "check_channel_order",
    "create_spatial_split",
    "SpatialSplit",
    "TRAIN",
    "VAL",
    "TEST",
    "OUTSIDE",
    "generate_patch_index",
    "PatchIndex",
    "patch_starts",
    "compute_terrain_mean",
    "save_terrain_mean",
    "load_terrain_mean",
]