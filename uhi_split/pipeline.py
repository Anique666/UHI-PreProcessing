"""Top-level orchestration for the UHI lazy-dataset stage.

Builds the lightweight, reproducible dataset artifact:

    processed_dataset/
    ├── sample_index.jsonl      deterministic per-sample records
    ├── patch_index.json        accepted patches + purity
    ├── month_catalog.json      explicit month -> path mapping
    ├── normalization_stats.json
    ├── split_config.json
    ├── metadata.json
    ├── terrain_mean.tif        precomputed static-terrain temporal mean
    ├── dataset.py              authoritative UHIDataset (copied verbatim)
    ├── README.md               comprehensive project context
    └── diagnostics/

It does NOT materialize X/y/mask tensors.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch

from .config import Config, load_config, dump_json
from .months import discover_months, inspect_grid, check_channel_order
from .patches import generate_patch_index_with_stats
from .split import create_spatial_split, SPLIT_NAMES, TRAIN, VAL, TEST
from .stats import compute_train_stats
from .terrain import compute_terrain_mean, save_terrain_mean
from .index import build_and_save_index, load_sample_index
from . import viz

HERE = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_geopandas_boundary(cfg: Config):
    import geopandas as gpd
    from shapely import make_valid

    g = gpd.read_file(cfg.paths.bbmp_boundary)
    g["geometry"] = g.geometry.apply(make_valid)
    return g.to_crs("EPSG:32643")


def _fresh_output(out_dir: Path) -> None:
    """Remove any previous build (including accidental materialized tensors)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def run(config_path: str, mode: str, test_months: int) -> dict:
    t0 = time.time()
    cfg = load_config(config_path)
    out_dir = cfg.paths.output_dir
    _fresh_output(out_dir)
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    _log("=" * 78)
    _log(f"UHI lazy-dataset build | mode={mode} | output={out_dir}")
    _log("=" * 78)

    # --- 1. Discovery & chronological ordering of ALL months -----------------
    all_months = discover_months(cfg.paths.input_dir)
    _log("[discovery]")
    _log(f"  number of TIFFs        : {len(all_months)}")
    _log(f"  first month            : {all_months[0].label}")
    _log(f"  last month             : {all_months[-1].label}")

    # --- 2. Grid / metadata validation ---------------------------------------
    grid = inspect_grid(all_months)
    check_channel_order(grid, cfg.data.channel_names)
    _log(f"  spatial dimensions     : {grid.width} x {grid.height} (W x H)")
    _log(f"  CRS                    : {grid.crs}")
    _log(f"  transform              : {grid.transform}")
    _log(f"  channel count          : {grid.count}")
    _log(f"  dtype                  : {grid.dtype}")
    _log("  band order             : OK (matches configured 18-channel order)")

    if mode == "test":
        n_months = max(test_months, cfg.data.sequence_length + 1)
        months = all_months[:n_months]
        _log(f"[test] index over first {len(months)} months "
             f"({months[0].label} -> {months[-1].label})")
    else:
        months = all_months
        _log(f"[full] index over all {len(months)} months")

    # --- 3. Spatial split ----------------------------------------------------
    split = create_spatial_split(grid, cfg.paths.bbmp_boundary, cfg.split)
    split.save(diag)
    pct = split.stats["pixel_pct"]
    _log("[spatial split]")
    for k in ("train", "val", "test"):
        _log(f"  {k:5s} pixel %: {pct[k]:6.2f}   pixels: {split.stats['pixels'][k]}")
    _log(f"  region area: {split.stats['region_area_km2']:.1f} km^2 over "
         f"{split.stats['region_pixels']} pixels")

    # --- 4. Precompute static-terrain temporal mean (exact) ------------------
    _log("[terrain] computing per-pixel temporal mean of static channels ...")
    terrain_arr = compute_terrain_mean(months, grid, cfg.data)
    save_terrain_mean(terrain_arr, grid, cfg.data, out_dir / "terrain_mean.tif")
    _log(f"  written: {out_dir / 'terrain_mean.tif'} "
         f"(bands={list(cfg.data.terrain_channels)})")

    # --- 5. Patches + purity assignment --------------------------------------
    patch_index, patch_stats = generate_patch_index_with_stats(
        split.split_mask, cfg.data.patch_size, cfg.data.stride,
        cfg.data.purity_threshold,
    )
    from collections import Counter
    pc = Counter(p.split for p in patch_index)
    _log("[patches]")
    _log(f"  patch_size={cfg.data.patch_size} stride={cfg.data.stride} "
         f"purity>={cfg.data.purity_threshold}")
    _log(f"  candidate={patch_stats['candidate_patches']} "
         f"accepted={patch_stats['accepted_patches']} "
         f"rejected={patch_stats['rejected_patches']} "
         f"({100*patch_stats['rejected_fraction']:.1f}%)")
    _log(f"  patches by split: {dict(pc)}")

    # --- 6. TRAIN-only fill + normalization statistics -----------------------
    _log("[statistics] computing TRAIN-only fill + normalization stats ...")
    stats = compute_train_stats(
        months, [p for p in patch_index if p.split == "train"],
        split.split_mask, cfg.data, terrain_arr,
        max_gap_fraction=cfg.data.max_gap_fraction,
    )
    dump_json(stats.as_dict(), out_dir / "normalization_stats.json")
    _log(f"  written: {out_dir / 'normalization_stats.json'}")
    if stats.degenerate_channels:
        _log(f"  WARNING degenerate channels (std~0 -> std:=1): "
             f"{stats.degenerate_channels}")
    if stats.over_gap_channels:
        _log(f"  WARNING channels with >{cfg.data.max_gap_fraction:.0%} "
             f"train missingness: {stats.over_gap_channels}")

    # --- 7. Deterministic index ----------------------------------------------
    _log("[index] building sample_index.jsonl + patch_index.json ...")
    idx_summary = build_and_save_index(cfg, months, patch_index, out_dir)
    _log(f"  patches: {idx_summary['patch_counts']}")
    _log(f"  samples: {idx_summary['sample_counts']}  "
         f"total={idx_summary['n_samples']}")
    _log(f"  temporal positions: {idx_summary['temporal_positions']}")
    _log(f"  written: {out_dir / 'sample_index.jsonl'}, "
         f"{out_dir / 'patch_index.json'}, {out_dir / 'month_catalog.json'}")
    if mode == "test":
        n_seq_all = len(all_months) - cfg.data.sequence_length
        proj = {k: idx_summary["patch_counts"][k] * n_seq_all
                for k in ("train", "val", "test")}
        _log(f"  PROJECTED full-run samples (72 months): {proj} "
             f"total={sum(proj.values())}")
    else:
        tot = idx_summary["n_samples"]
        _log(f"  sample split %: " + ", ".join(
            f"{k}={100*v/tot:.1f}%" for k, v in
            idx_summary["sample_counts"].items()))

    # --- 8. Metadata + copy authoritative implementation ---------------------
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "architecture": "lazy dataset (windowed TIFF reads; no materialized tensors)",
        "config": cfg.to_dict(),
        "months_used": [m.label for m in months],
        "tensor_shapes": {
            "X": f"[12, {len(cfg.data.channel_names)}, "
                 f"{cfg.data.patch_size}, {cfg.data.patch_size}]",
            "y": f"[1, {cfg.data.patch_size}, {cfg.data.patch_size}]",
            "mask": f"[1, {cfg.data.patch_size}, {cfg.data.patch_size}]",
            "batched_X": f"[B, 12, {len(cfg.data.channel_names)}, "
                          f"{cfg.data.patch_size}, {cfg.data.patch_size}]",
        },
        "split_stats": split.stats,
        "patch_stats": patch_stats,
        "index_summary": idx_summary,
        "terrain_treatment": (
            "per-pixel temporal mean over the run's months, applied to every "
            "timestep; precomputed in terrain_mean.tif (identical to the "
            "validated per-run terrain nanmean)"
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "rasterio": rasterio.__version__,
            "torch": torch.__version__,
        },
    }
    dump_json(meta, out_dir / "metadata.json")
    dump_json(split.stats, out_dir / "split_config.json")
    # Re-emit the effective config actually used (path + hyper-parameters).
    try:
        import yaml
        with open(out_dir / "_effective_config.yaml", "w") as f:
            yaml.safe_dump(
                {
                    "paths": {
                        "input_dir": str(cfg.paths.input_dir),
                        "output_dir": str(cfg.paths.output_dir),
                        "bbmp_boundary": str(cfg.paths.bbmp_boundary),
                    },
                    "data": cfg.data.__dict__,
                    "split": cfg.split.__dict__,
                },
                f,
                sort_keys=False,
            )
    except Exception as exc:  # pragma: no cover - non-fatal
        _log(f"  WARNING could not write _effective_config.yaml: {exc!r}")
    shutil.copy2(HERE / "dataset.py", out_dir / "dataset.py")
    readme_src = HERE / "PROJECT_README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, out_dir / "README.md")
    _log(f"[artifacts] copied dataset.py and README.md into {out_dir}")

    # --- 9. Split visualizations ---------------------------------------------
    boundary = _load_geopandas_boundary(cfg)
    bounds = (
        grid.transform[0],
        grid.transform[0] + grid.width * grid.transform[1],
        grid.transform[3] + grid.height * grid.transform[5],
        grid.transform[3],
    )
    viz.plot_spatial_split(
        split.split_mask, split.region_mask, split.stats,
        diag / "spatial_split_map.png", boundary=boundary, bounds=bounds,
    )
    viz.plot_split_profiles(split.split_mask, diag / "split_profiles.png")

    # --- 10. Verify lazy samples ---------------------------------------------
    verify = verify_dataset(cfg, out_dir, months, split, stats, diag)
    dump_json(verify, diag / "lazy_validation.json")
    _log("[verify] lazy UHIDataset")
    for split_name, info in verify["splits"].items():
        _log(f"  {split_name:5s}: n={info['n_samples']} "
             f"X{info['X_shape']} y{info['y_shape']} mask{info['mask_shape']} "
             f"NaN={info['nan']} Inf={info['inf']} "
             f"mask_valid_frac={info['mask_valid_frac']:.4f}")
    _log(f"  temporal examples: {len(verify['temporal_examples'])}")
    if "sample_plot" in verify:
        _log(f"  sample plot: {verify['sample_plot']}")
    _log(f"[done] elapsed {time.time() - t0:.1f}s")
    return verify


def verify_dataset(cfg, out_dir, months, split, stats, diag) -> dict:
    """Load several samples per split through UHIDataset and validate them,
    including a raw-data cross-check of the target LST and temporal labels."""
    from .dataset import UHIDataset

    report: dict = {"splits": {}, "checks": [], "temporal_examples": []}
    lst_idx = cfg.data.channel_names.index(cfg.data.lst_channel)
    terrain_idx = [cfg.data.channel_names.index(c) for c in cfg.data.terrain_channels]

    ds_by_split = {}
    for split_name in ("train", "val", "test"):
        ds = UHIDataset(out_dir, split=split_name)
        ds_by_split[split_name] = ds
        n = len(ds)
        # deterministic picks: first, middle, last
        picks = sorted({0, n // 2, n - 1})
        nan_total = inf_total = 0
        mask_valid = []
        shapes = set()
        for i in picks:
            X, y, mask = ds[i]
            shapes.add((tuple(X.shape), tuple(y.shape), tuple(mask.shape)))
            nan_total += int(torch.isnan(X).sum() + torch.isnan(y).sum())
            inf_total += int(torch.isinf(X).sum() + torch.isinf(y).sum())
            mask_valid.append(float(mask.float().mean()))
        if len(shapes) != 1:
            raise ValueError(f"Inconsistent sample shapes in {split_name}: {shapes}")
        Xs, ys, ms = next(iter(shapes))
        report["splits"][split_name] = {
            "n_samples": n,
            "X_shape": list(Xs), "y_shape": list(ys), "mask_shape": list(ms),
            "X_dtype": "float32", "y_dtype": "float32",
            "nan": nan_total, "inf": inf_total,
            "mask_valid_frac": float(np.mean(mask_valid)),
        }

    # Raw cross-check + temporal verification for a train and a test sample.
    for split_name in ("train", "test"):
        ds = ds_by_split[split_name]
        rec = ds.sample_record(0)
        X, y, mask = ds[0]
        # Reconstruct expected LST directly from TIFFs (independent path).
        inp = rec["input_months"][-1]
        mpath = ds._month_path(inp)
        with rasterio.open(mpath) as rd:
            raw = rd.read(lst_idx + 1, window=(
                (rec["row_off"], rec["row_off"] + rec["height"]),
                (rec["col_off"], rec["col_off"] + rec["width"]),
            )).astype(np.float32)
        exp = (raw - stats.means[lst_idx]) / stats.stds[lst_idx]
        finite = np.isfinite(raw)
        xlast = X[-1, lst_idx].numpy()
        maxdiff = float(np.abs(xlast[finite] - exp[finite]).max())
        # target
        tpath = ds._month_path(rec["target_month"])
        with rasterio.open(tpath) as rd:
            raw_t = rd.read(lst_idx + 1, window=(
                (rec["row_off"], rec["row_off"] + rec["height"]),
                (rec["col_off"], rec["col_off"] + rec["width"]),
            )).astype(np.float32)
        exp_y = (raw_t - stats.means[lst_idx]) / stats.stds[lst_idx]
        tv = np.isfinite(raw_t)
        report["checks"].append({
            "split": split_name,
            "sample_id": rec["sample_id"],
            "input_month_checked": inp,
            "target_month": rec["target_month"],
            "lst_input_max_abs_diff_vs_raw": maxdiff,
            "target_mask_matches_raw_finite": bool(
                np.array_equal(mask[0].numpy().astype(bool), tv)
            ),
            "target_y_max_abs_diff_vs_raw": float(
                np.abs(y[0].numpy()[tv] - exp_y[tv]).max()
            ) if tv.any() else 0.0,
        })
        report["temporal_examples"].append({
            "split": split_name,
            "sample_id": rec["sample_id"],
            "input_months": rec["input_months"],
            "target_month": rec["target_month"],
        })

    # Diagnostic plot: historical LST vs target LST vs mask for one train sample.
    try:
        ds = ds_by_split["train"]
        X, y, mask = ds[0]
        rec = ds.sample_record(0)
        viz.plot_sample_grid(
            X.numpy(), y.numpy(), mask.numpy(), ds.channel_names, lst_idx,
            diag / "sample_lst_patch.png", rec["target_month"],
            rec["input_months"],
        )
        report["sample_plot"] = str(diag / "sample_lst_patch.png")
    except Exception as exc:  # diagnostics must not hide real errors
        report["sample_plot_error"] = repr(exc)

    for ds in ds_by_split.values():
        ds.close()
    return report