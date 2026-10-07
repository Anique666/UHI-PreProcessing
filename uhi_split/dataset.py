"""Authoritative lazy-loading PyTorch Dataset for the Bengaluru UHI project.

This file is self-contained on purpose: it can be copied verbatim into a
processed-dataset directory (``processed_dataset/dataset.py``) and used without
importing the preprocessing package. The preprocessing pipeline copies this
exact file to the output, so there is only ONE implementation.

It reconstructs individual samples on demand from the canonical monthly
GeoTIFFs using windowed reads, then applies the previously validated TRAIN-only
fill + normalization. No materialized X/y chunk files are used.

Sample reconstruction (per index):
    sample index -> 12 historical month labels + target month + patch window
                 -> windowed reads of the required 128x128 region
                 -> terrain channel replaced by the precomputed temporal mean
                 -> fill NaN/Inf with TRAIN fill values
                 -> normalize with TRAIN mean/std
                 -> X [12,18,128,128], y [1,128,128], mask [1,128,128]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

__all__ = ["UHIDataset"]


class _ReaderCache:
    """Per-process cache of open rasterio datasets.

    Datasets are opened lazily on first use (i.e. after DataLoader workers have
    forked), which keeps it safe with num_workers>0 / persistent_workers=True.
    """

    def __init__(self, max_open: int = 24):
        self._open: dict[str, rasterio.DatasetReader] = {}
        self._order: list[str] = []
        self.max_open = max_open

    def reader(self, path: str) -> rasterio.DatasetReader:
        ds = self._open.get(path)
        if ds is None:
            ds = rasterio.open(path)
            self._open[path] = ds
            self._order.append(path)
            while len(self._order) > self.max_open:
                old = self._order.pop(0)
                if old in self._open:
                    self._open.pop(old).close()
        return ds

    def close(self) -> None:
        for ds in self._open.values():
            try:
                ds.close()
            except Exception:
                pass
        self._open.clear()
        self._order.clear()

    def __del__(self):  # pragma: no cover - best effort
        self.close()


class UHIDataset(Dataset):
    """Lazy UHI dataset.

    Parameters
    ----------
    dataset_root : str | Path
        Directory containing ``sample_index.jsonl``, ``patch_index.json``,
        ``month_catalog.json``, ``normalization_stats.json`` and
        ``terrain_mean.tif``.
    split : {"train", "val", "test", None}
        Restrict to one split (None = all samples, ordered by sample_id).
    input_dir : str | Path | None
        Override the TIFF root saved in ``month_catalog.json`` (useful if the
        processed TIFFs are mounted elsewhere).
    max_open_rasters : int
        Reader-cache size.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str | None = None,
        input_dir: str | Path | None = None,
        max_open_rasters: int = 24,
    ):
        self.root = Path(dataset_root)
        if split not in (None, "train", "val", "test"):
            raise ValueError(f"Invalid split: {split!r}")
        self.split = split

        with open(self.root / "normalization_stats.json") as f:
            stats = json.load(f)
        self.channel_names: list[str] = list(stats["channel_names"])
        self.fill_values = np.asarray(stats["fill_values"], dtype=np.float32)
        self.means = np.asarray(stats["train_means"], dtype=np.float32)
        self.stds = np.asarray(stats["train_stds"], dtype=np.float32)
        if not (len(self.channel_names) == len(self.fill_values) ==
                len(self.means) == len(self.stds)):
            raise ValueError("normalization_stats.json arrays have mismatched lengths")
        if np.any(self.stds == 0):
            raise ValueError("normalization_stats.json contains a zero std")

        with open(self.root / "month_catalog.json") as f:
            cat = json.load(f)
        self._saved_input_dir = Path(cat["input_dir"])
        self._input_dir = Path(input_dir) if input_dir else self._saved_input_dir
        self.month_labels: list[str] = [m["label"] for m in cat["months"]]
        self._month_index = {m["label"]: int(m["index"]) for m in cat["months"]}
        self._month_relpath = {
            m["label"]: m["relative_path"] for m in cat["months"]
        }

        with open(self.root / "patch_index.json") as f:
            pdoc = json.load(f)
        self.patch_size = int(pdoc["patch_size"])
        self.stride = int(pdoc["stride"])
        self.purity_threshold = float(pdoc["purity_threshold"])

        # sample records (deterministic order by sample_id)
        records = []
        with open(self.root / "sample_index.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda r: r["sample_id"])
        if split is not None:
            records = [r for r in records if r["split"] == split]
        self.records = records
        if not self.records:
            raise ValueError(f"No samples found for split={split!r}")

        self.terrain_mean_path = self.root / "terrain_mean.tif"
        self._terrain_idx = [
            self.channel_names.index(c)
            for c in ["Elevation", "Slope"]
            if c in self.channel_names
        ]
        self._lst_idx = self.channel_names.index("LST_Celsius")
        self._n_ch = len(self.channel_names)
        self._readers = _ReaderCache(max_open=max_open_rasters)

    # -- helpers -----------------------------------------------------------
    def _month_path(self, label: str) -> str:
        rel = self._month_relpath[label]
        p = Path(rel)
        if p.is_absolute():
            return str(p)
        return str(self._input_dir / p)

    def __len__(self) -> int:
        return len(self.records)

    def sample_record(self, idx: int) -> dict:
        return self.records[idx]

    # -- core --------------------------------------------------------------
    def __getitem__(self, idx: int):
        rec = self.records[idx]
        r0, c0 = int(rec["row_off"]), int(rec["col_off"])
        s = int(rec["height"])
        if s != int(rec["width"]):
            raise ValueError(f"Non-square patch in sample {rec['sample_id']}")
        window = ((r0, r0 + s), (c0, c0 + s))

        input_labels = rec["input_months"]
        C = self._n_ch
        X = np.empty((len(input_labels), C, s, s), dtype=np.float32)
        dynamic = [c for c in range(C) if c not in self._terrain_idx]

        for t, label in enumerate(input_labels):
            ds = self._readers.reader(self._month_path(label))
            # Read only dynamic bands; terrain is overwritten below.
            win = ds.read(
                indexes=[c + 1 for c in dynamic], window=window
            ).astype(np.float32, copy=True)
            for j, c in enumerate(dynamic):
                X[t, c] = win[j]

        # Terrain: replace with precomputed per-pixel temporal mean (identical
        # to the validated per-run nanmean), applied to every timestep.
        if self._terrain_idx:
            tds = self._readers.reader(str(self.terrain_mean_path))
            tw = tds.read(window=window).astype(np.float32, copy=True)
            for j, c in enumerate(self._terrain_idx):
                X[:, c] = tw[j][None, :, :]

        # Fill + normalize inputs (train-only statistics).
        fills = self.fill_values[None, :, None, None]
        means = self.means[None, :, None, None]
        stds = self.stds[None, :, None, None]
        Xf = np.where(np.isfinite(X), X, fills)
        Xn = (Xf - means) / stds

        # Target LST for the following month.
        tgt_label = rec["target_month"]
        tds = self._readers.reader(self._month_path(tgt_label))
        raw_t = tds.read(self._lst_idx + 1, window=window).astype(np.float32)
        mask = np.isfinite(raw_t)
        li = self._lst_idx
        yf = np.where(mask, raw_t, float(self.fill_values[li]))
        yn = (yf - float(self.means[li])) / float(self.stds[li])

        X_t = torch.from_numpy(np.ascontiguousarray(Xn, dtype=np.float32))
        y_t = torch.from_numpy(np.ascontiguousarray(yn[None], dtype=np.float32))
        m_t = torch.from_numpy(np.ascontiguousarray(mask[None]))
        return X_t, y_t, m_t

    def close(self) -> None:
        self._readers.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_readers"] = None  # never pickle open file handles
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._readers = _ReaderCache()


# ---------------------------------------------------------------------------
# Minimal usage example (also used by tests / the training stage).
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="UHIDataset demo")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    dataset = UHIDataset(args.dataset_root, split=args.split)
    X, y, mask = dataset[0]
    print("dataset[0]")
    print("  X   ", tuple(X.shape), X.dtype)
    print("  y   ", tuple(y.shape), y.dtype)
    print("  mask", tuple(mask.shape), mask.dtype)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    Xb, yb, mb = next(iter(loader))
    print("DataLoader batch")
    print("  X   ", tuple(Xb.shape))
    print("  y   ", tuple(yb.shape))
    print("  mask", tuple(mb.shape))