"""Deterministic sample/patch index construction.

Everything a sample needs to be reconstructed is written explicitly:
split, spatial window (row_off/col_off/height/width), temporal window
(input month labels + target month) and stable integer ids. Nothing relies on
implicit filesystem ordering or on filtered-list positions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .config import Config
from .months import MonthRecord
from .patches import PatchIndex
from .split import TRAIN, VAL, TEST

_SPLIT_ID = {"train": TRAIN, "val": VAL, "test": TEST}


@dataclass
class PatchRecord:
    patch_id: int
    split: str
    row_off: int
    col_off: int
    height: int
    width: int
    purity: float
    pixel_counts: dict


@dataclass
class SampleRecord:
    sample_id: int
    split: str
    patch_id: int
    row_off: int
    col_off: int
    height: int
    width: int
    seq_start_index: int
    input_month_indices: list[int]
    target_month_index: int
    input_months: list[str]
    target_month: str


def build_patch_records(patch_index: list[PatchIndex]) -> list[PatchRecord]:
    recs: list[PatchRecord] = []
    for pid, pi in enumerate(patch_index):
        purity = pi.counts[pi.split] / (pi.size * pi.size)
        recs.append(
            PatchRecord(
                patch_id=pid,
                split=pi.split,
                row_off=pi.row0,
                col_off=pi.col0,
                height=pi.size,
                width=pi.size,
                purity=round(purity, 6),
                pixel_counts=pi.counts,
            )
        )
    return recs


def build_sample_records(
    months: list[MonthRecord],
    patch_records: list[PatchRecord],
    sequence_length: int,
) -> list[SampleRecord]:
    n_seq = len(months) - sequence_length
    if n_seq <= 0:
        raise ValueError(
            f"Not enough months ({len(months)}) for sequence_length "
            f"{sequence_length}"
        )
    samples: list[SampleRecord] = []
    sid = 0
    # patches in patch_id order; all temporal positions in chronological order.
    for pr in patch_records:
        for i in range(n_seq):
            in_idx = list(range(i, i + sequence_length))
            tgt = i + sequence_length
            samples.append(
                SampleRecord(
                    sample_id=sid,
                    split=pr.split,
                    patch_id=pr.patch_id,
                    row_off=pr.row_off,
                    col_off=pr.col_off,
                    height=pr.height,
                    width=pr.width,
                    seq_start_index=i,
                    input_month_indices=in_idx,
                    target_month_index=tgt,
                    input_months=[months[k].label for k in in_idx],
                    target_month=months[tgt].label,
                )
            )
            sid += 1
    return samples


def build_and_save_index(
    cfg: Config,
    months: list[MonthRecord],
    patch_index: list[PatchIndex],
    out_dir: Path,
) -> dict:
    patch_records = build_patch_records(patch_index)
    sample_records = build_sample_records(
        months, patch_records, cfg.data.sequence_length
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # sample_index.jsonl (one record per line, deterministic order by sample_id)
    with open(out_dir / "sample_index.jsonl", "w") as f:
        for s in sample_records:
            f.write(json.dumps(asdict(s)) + "\n")

    # patch_index.json
    patch_doc = {
        "patch_size": cfg.data.patch_size,
        "stride": cfg.data.stride,
        "purity_threshold": cfg.data.purity_threshold,
        "n_patches": len(patch_records),
        "patches": [asdict(p) for p in patch_records],
    }
    with open(out_dir / "patch_index.json", "w") as f:
        json.dump(patch_doc, f, indent=2)

    # month catalog: explicit index -> label -> relative path (no fs ordering)
    input_root = cfg.paths.input_dir
    catalog = []
    for i, m in enumerate(months):
        try:
            rel = str(m.path.relative_to(input_root))
        except ValueError:
            rel = str(m.path)
        catalog.append({"index": i, "label": m.label, "relative_path": rel})
    with open(out_dir / "month_catalog.json", "w") as f:
        json.dump(
            {
                "input_dir": str(input_root),
                "n_months": len(catalog),
                "months": catalog,
            },
            f,
            indent=2,
        )

    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = sum(1 for s in sample_records if s.split == split)
    patch_counts = {}
    for split in ("train", "val", "test"):
        patch_counts[split] = sum(1 for p in patch_records if p.split == split)

    return {
        "n_samples": len(sample_records),
        "n_patches": len(patch_records),
        "sample_counts": counts,
        "patch_counts": patch_counts,
        "sequence_length": cfg.data.sequence_length,
        "temporal_positions": len(months) - cfg.data.sequence_length,
    }


def load_sample_index(path: Path, split: str | None = None) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split is None or rec["split"] == split:
                out.append(rec)
    # enforce deterministic order even if the file was edited by hand
    out.sort(key=lambda r: r["sample_id"])
    return out