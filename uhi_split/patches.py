"""Candidate 128x128 patch enumeration and split assignment via a purity rule.

A patch is assigned to a split only if at least `purity_threshold` (default
90%) of its pixels belong to that split. Pixels outside the BBMP modeling area
belong to no split and therefore count against purity. A patch that satisfies
no split is rejected (never reassigned). Because a pixel has exactly one split
id, at most one split can reach the purity threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .split import TRAIN, VAL, TEST, SPLIT_NAMES


@dataclass
class PatchIndex:
    split: str
    row0: int
    col0: int
    size: int
    counts: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "row0": self.row0,
            "col0": self.col0,
            "size": self.size,
            "pixel_counts": self.counts,
        }


def patch_starts(length: int, size: int, stride: int) -> list[int]:
    """Top-left offsets that keep the window fully inside the raster."""
    if length < size:
        return []
    last = length - size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _assign(tile: np.ndarray, threshold: float) -> tuple[str | None, dict[str, int]]:
    size = tile.size
    needed = math.ceil(threshold * size)
    counts = {name: int((tile == sid).sum()) for sid, name in SPLIT_NAMES.items()}
    winners = [name for name, c in counts.items() if c >= needed]
    if not winners:
        return None, counts
    if len(winners) > 1:  # impossible with exclusive pixel ids; guard loudly
        raise AssertionError(f"Multiple splits passed purity: {winners} / {counts}")
    return winners[0], counts


def generate_patch_index(
    split_mask: np.ndarray, size: int, stride: int, threshold: float
) -> list[PatchIndex]:
    accepted, _ = generate_patch_index_with_stats(split_mask, size, stride, threshold)
    return accepted


def generate_patch_index_with_stats(
    split_mask: np.ndarray, size: int, stride: int, threshold: float
) -> tuple[list[PatchIndex], dict]:
    H, W = split_mask.shape
    rows = patch_starts(H, size, stride)
    cols = patch_starts(W, size, stride)
    out: list[PatchIndex] = []
    n_candidates = 0
    n_rejected = 0
    rejected_purity_hist: list[float] = []
    for r in rows:
        for c in cols:
            n_candidates += 1
            tile = split_mask[r : r + size, c : c + size]
            name, counts = _assign(tile, threshold)
            if name is None:
                n_rejected += 1
                best = max((c2 / tile.size) for c2 in counts.values())
                rejected_purity_hist.append(best)
                continue
            out.append(PatchIndex(name, r, c, size, counts))
    if not out:
        raise ValueError(
            "No patch passed the purity threshold. Check the spatial split, the "
            "patch size/stride and the purity threshold."
        )
    stats = {
        "candidate_patches": n_candidates,
        "accepted_patches": len(out),
        "rejected_patches": n_rejected,
        "rejected_fraction": n_rejected / n_candidates if n_candidates else 0.0,
        "rejected_best_split_purity": {
            "min": float(np.min(rejected_purity_hist)) if rejected_purity_hist else None,
            "max": float(np.max(rejected_purity_hist)) if rejected_purity_hist else None,
            "mean": float(np.mean(rejected_purity_hist)) if rejected_purity_hist else None,
        },
    }
    return out, stats