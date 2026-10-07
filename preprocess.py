#!/usr/bin/env python3
"""Standalone UHI preprocessing stage: spatial split, train-only normalization,
12-month temporal sequences and a lightweight LAZY PyTorch Dataset.

This DOES NOT modify or rerun the previous temporal interpolation pipeline; it
only consumes its processed monthly TIFFs. It also does NOT materialize X/y
tensors: it writes deterministic index/metadata files plus the authoritative
``dataset.py`` (UHIDataset), which reconstructs samples on demand via windowed
TIFF reads.

Usage
-----
Test (small subset; writes real lightweight metadata + diagnostics, validates
the lazy Dataset):
    python preprocess.py --mode test

Full (all 72 months; builds the complete index and final TRAIN-only statistics;
still writes NO tensors):
    python preprocess.py --mode full

All paths and hyper-parameters are configurable via --config and CLI overrides.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from uhi_split.config import load_config          # noqa: E402
from uhi_split.pipeline import run                # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(HERE / "uhi_split" / "preprocess_config.yaml"),
                   help="YAML config path")
    p.add_argument("--input", default=None, help="override input_dir")
    p.add_argument("--output", default=None, help="override output_dir")
    p.add_argument("--mode", choices=["test", "full"], default="test",
                   help="test = small subset exercising all stages; full = all 72 months")
    p.add_argument("--test-months", type=int, default=13,
                   help="number of earliest months to index in test mode")
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--purity", type=float, default=None, help="purity threshold (0,1]")
    p.add_argument("--sequence-length", type=int, default=None)
    p.add_argument("--max-open-rasters", type=int, default=None,
                   help="per-worker raster reader cache size")
    p.add_argument("--rotation-deg", type=float, default=None,
                   help="rotate the spatial partition axes")
    return p


def apply_overrides(cfg_path: str, args) -> str:
    """If CLI overrides are given, materialise an effective config file."""
    cfg = load_config(cfg_path)
    changed = False
    if args.input:
        cfg.paths.input_dir = Path(args.input)
        changed = True
    if args.output:
        cfg.paths.output_dir = Path(args.output)
        changed = True
    if args.patch_size is not None:
        cfg.data.patch_size = args.patch_size
        changed = True
    if args.stride is not None:
        cfg.data.stride = args.stride
        changed = True
    if args.purity is not None:
        cfg.data.purity_threshold = args.purity
        changed = True
    if args.sequence_length is not None:
        cfg.data.sequence_length = args.sequence_length
        changed = True
    if args.max_open_rasters is not None:
        cfg.data.max_open_rasters = args.max_open_rasters
        changed = True
    if args.rotation_deg is not None:
        cfg.split.rotation_deg = args.rotation_deg
        changed = True
    if not changed:
        return cfg_path

    import tempfile
    import yaml
    eff = {
        "paths": {
            "input_dir": str(cfg.paths.input_dir),
            "output_dir": str(cfg.paths.output_dir),
            "bbmp_boundary": str(cfg.paths.bbmp_boundary),
        },
        "data": cfg.data.__dict__,
        "split": cfg.split.__dict__,
    }
    # Write to a temp path: the pipeline wipes output_dir at the start, then
    # re-emits the effective config inside it.
    fd, tmp = tempfile.mkstemp(prefix="uhi_effective_", suffix=".yaml")
    with open(fd, "w") as f:
        yaml.safe_dump(eff, f, sort_keys=False)
    print(f"[cli] effective config (temp): {tmp}", flush=True)
    return tmp


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = apply_overrides(args.config, args)
    report = run(cfg_path, mode=args.mode, test_months=args.test_months)

    counts = report["splits"]
    total = sum(v["n_samples"] for v in counts.values())
    nan = sum(v["nan"] for v in counts.values())
    inf = sum(v["inf"] for v in counts.values())
    print("\n=== FINAL SUMMARY (lazy Dataset) ===")
    print(f"total samples  : {total}")
    for k, v in counts.items():
        print(f"  {k:5s}: n={v['n_samples']:5d}  "
              f"X{tuple(v['X_shape'])} y{tuple(v['y_shape'])} "
              f"mask{tuple(v['mask_shape'])}")
    print(f"NaN count      : {nan}")
    print(f"Inf count      : {inf}")
    print("No X/y tensors were materialized (lazy architecture).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())