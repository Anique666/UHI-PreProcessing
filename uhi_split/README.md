# UHI preprocessing stage (lazy dataset)

This directory builds a **lightweight, reproducible dataset artifact** from the
already-interpolated monthly TIFFs. It does **not** materialize X/y tensors:
the canonical monthly TIFFs remain the data source and `UHIDataset` reconstructs
samples on demand via windowed reads.

See **`PROJECT_README.md`** in this directory for the full scientific and
methodological project context (dataset, channels, temporal formulation, spatial
split, missing-data handling, normalization, patch rules, validated results,
open decisions, and research-paper notes). It is copied to
`processed_dataset/README.md` when the dataset is built.

## Commands

```bash
# Test: first 24 months; builds full lightweight metadata + validates the
# lazy Dataset; writes NO tensors.
python preprocess.py \
    --input  /path/to/processed_temporal \
    --output /path/to/processed_dataset \
    --mode test

# Full: all 72 months; builds the complete index and final TRAIN-only
# statistics; still writes NO tensors.
python preprocess.py \
    --input  /path/to/processed_temporal \
    --output /path/to/processed_dataset \
    --mode full
```

## Verifying a built dataset

```bash
python verify_dataset.py --dataset-root /path/to/processed_dataset
```

## Using the Dataset

```python
from dataset import UHIDataset          # copied into processed_dataset/
from torch.utils.data import DataLoader

ds = UHIDataset("processed_dataset", split="train")
X, y, mask = ds[0]                       # [12,18,128,128] / [1,128,128] / [1,128,128]

loader = DataLoader(ds, batch_size=8, num_workers=4,
                    pin_memory=True, persistent_workers=True)
Xb, yb, mb = next(iter(loader))          # [8,12,18,128,128] / ...
```

## Layout

```
uhi_split/
├── preprocess_config.yaml   # all hyper-parameters
├── config.py                # config load + validation
├── months.py                # discovery, chronology, grid validation
├── split.py                 # spatial partition + masks
├── patches.py               # patch enumeration + purity
├── stats.py                 # TRAIN-only fill + normalization stats
├── terrain.py               # static-terrain temporal-mean precompute
├── index.py                 # deterministic sample/patch/month index
├── dataset.py               # AUTHORITATIVE UHIDataset (copied to output)
├── viz.py                   # diagnostic plots
├── pipeline.py              # orchestration + lazy validation
├── preprocess.py (../)      # CLI entry point
├── verify_dataset.py (../)  # standalone verifier
└── PROJECT_README.md        # full project context
```

The authoritative Dataset implementation is `dataset.py`; the pipeline copies it
verbatim into the output so there is only one codebase.