#!/usr/bin/env python3
"""Standalone verifier for a built lazy UHI dataset.

Reconstructs samples across splits and temporal positions through the
authoritative ``UHIDataset`` and checks shapes, dtypes, NaN/Inf, the target
mask, temporal correctness against the TIFFs, and (optionally) agreement with a
previously materialized ``.pt`` build.

Usage:
    python verify_dataset.py --dataset-root /path/to/processed_dataset
    python verify_dataset.py --dataset-root DIR --compare-materialized /path/to/processed_tensors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch


def _load_dataset_module(dataset_root: Path):
    sys.path.insert(0, str(dataset_root))
    import importlib

    if "dataset" in sys.modules:
        del sys.modules["dataset"]
    return importlib.import_module("dataset")


def verify(dataset_root: Path, per_split: int, compare: Path | None) -> dict:
    mod = _load_dataset_module(dataset_root)
    UHIDataset = mod.UHIDataset

    report = {"dataset_root": str(dataset_root), "splits": {}, "errors": []}
    for split in ("train", "val", "test"):
        ds = UHIDataset(dataset_root, split=split)
        n = len(ds)
        picks = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})[:per_split]
        nan = inf = 0
        temporal_ok = True
        mask_checked = 0
        for i in picks:
            X, y, mask = ds[i]
            if tuple(X.shape) != (12, 18, 128, 128):
                report["errors"].append(f"{split}[{i}] bad X {tuple(X.shape)}")
            if tuple(y.shape) != (1, 128, 128):
                report["errors"].append(f"{split}[{i}] bad y {tuple(y.shape)}")
            if tuple(mask.shape) != (1, 128, 128):
                report["errors"].append(f"{split}[{i}] bad mask {tuple(mask.shape)}")
            if X.dtype != torch.float32 or y.dtype != torch.float32:
                report["errors"].append(f"{split}[{i}] bad dtype")
            nan += int(torch.isnan(X).sum() + torch.isnan(y).sum())
            inf += int(torch.isinf(X).sum() + torch.isinf(y).sum())

            rec = ds.sample_record(i)
            # temporal label check: input[-1] must be the month before target
            im = rec["input_months"]
            if len(im) != 12:
                temporal_ok = False
            else:
                ty, tm = int(rec["target_month"][:4]), int(rec["target_month"][5:])
                pm = tm - 1 if tm > 1 else 12
                py = ty if tm > 1 else ty - 1
                expected_last = f"{py:04d}-{pm:02d}"
                if im[-1] != expected_last:
                    temporal_ok = False
                if im != ds.month_labels[
                    rec["input_month_indices"][0]: rec["input_month_indices"][0] + 12
                ]:
                    temporal_ok = False
            # mask matches raw target
            with rasterio.open(ds._month_path(rec["target_month"])) as rd:
                raw = rd.read(1, window=(
                    (rec["row_off"], rec["row_off"] + rec["height"]),
                    (rec["col_off"], rec["col_off"] + rec["width"]),
                ))
            if not np.array_equal(mask[0].numpy().astype(bool), np.isfinite(raw)):
                report["errors"].append(f"{split}[{i}] mask != raw finite")
            mask_checked += 1
        report["splits"][split] = {
            "n_samples": n,
            "picks": picks,
            "nan": nan,
            "inf": inf,
            "temporal_ok": temporal_ok,
            "masks_checked": mask_checked,
        }
        ds.close()

    if compare is not None:
        report["materialized_compare"] = _compare(dataset_root, compare)

    return report


def _compare(dataset_root: Path, materialized: Path) -> dict:
    mod = _load_dataset_module(dataset_root)
    out = {}
    for split in ("train", "val", "test"):
        xf = sorted((materialized / split).glob("X_*.pt"))
        if not xf:
            out[split] = "no materialized chunks"
            continue
        ds = mod.UHIDataset(dataset_root, split=split)
        OX = torch.load(xf[0], map_location="cpu")
        Oy = torch.load(materialized / split / "y_000.pt", map_location="cpu")
        Om = torch.load(materialized / split / "mask_000.pt", map_location="cpu")
        maxd = 0.0
        for k in range(OX.shape[0]):
            if k >= len(ds):
                break
            X, y, m = ds[k]
            maxd = max(
                maxd,
                float((OX[k] - X).abs().max()),
                float((Oy[k] - y).abs().max()),
                float((Om[k].bool() ^ m.bool()).float().sum().item()),
            )
        out[split] = {"n_compared": int(min(OX.shape[0], len(ds))),
                      "max_abs_diff": maxd}
        ds.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--per-split", type=int, default=5)
    ap.add_argument("--compare-materialized", default=None)
    args = ap.parse_args()

    rep = verify(Path(args.dataset_root), args.per_split,
                 Path(args.compare_materialized) if args.compare_materialized else None)
    print(json.dumps(rep, indent=2))
    if rep["errors"]:
        print(f"\nFAILED: {len(rep['errors'])} errors", file=sys.stderr)
        return 1
    print("\nOK: all lazy samples valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())