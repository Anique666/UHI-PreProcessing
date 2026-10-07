# UHI-PreProcessing

Reproducible preprocessing and data-loading stage for a Bengaluru Urban Heat
Island (UHI) forecasting project. This repository turns a set of already
interpolated monthly feature GeoTIFFs into a **lazy PyTorch dataset** that
yields model-ready tensors for a ConvLSTM that predicts next-month land surface
temperature (LST).

This repository covers **preprocessing and data loading only**. It does **not**
contain model training, the ConvLSTM, or the later building-footprint
counterfactual/scenario engine.

```
already-interpolated monthly GeoTIFFs
        ↓
spatial train / validation / test split
        ↓
128 × 128 patches  (stride 64, ≥ 90% purity)
        ↓
12-month temporal sequences  →  next-month LST target
        ↓
TRAIN-only fill statistics
        ↓
TRAIN-only normalization
        ↓
deterministic sample index
        ↓
lazy UHIDataset  (windowed reads, no materialized tensors)
        ↓
[B, 12, 18, 128, 128]
```

---

## Input data

The pipeline consumes **already-preprocessed** monthly feature stacks. Temporal
interpolation was performed **upstream** by an earlier stage and is **not**
rerun here:

- 72 monthly GeoTIFFs, **January 2019 → December 2024**
- Each file, e.g. `Bengaluru_YYYY_MM_18features.tif`, contains 18 bands
- **1608 × 1493** pixels, **30 m** resolution (native, not resampled)
- CRS **EPSG:32643** (UTM zone 43N)
- `float32`, NaN used for missing data
- The band descriptions must match the 18-channel order below; the loader
  refuses to proceed otherwise

### Upstream temporal interpolation (already done)

- per-pixel, **linear**, **along the time axis**
- only **NaN/Inf** values were filled; valid observations were never changed
- **no spatial interpolation**, **no extrapolation**
- long gaps remain missing where appropriate

This repository consumes the result of that stage; it never modifies the raw
TIFFs and never reruns interpolation.

### The 18 channels (exact tensor order)

| # | Channel | Notes |
|--:|---|---|
| 1 | `LST_Celsius` | LST — also the prediction target |
| 2 | `NDVI` | |
| 3 | `NDBI` | |
| 4 | `MNDWI` | |
| 5 | `EVI` | |
| 6 | `Albedo` | |
| 7 | `Vegetation_Frac` | |
| 8 | `Impervious_Frac` | |
| 9 | `Soil_Frac` | |
| 10 | `Elevation` | static/near-static terrain |
| 11 | `Slope` | static/near-static terrain |
| 12 | `S2_B2` | Sentinel-2 blue |
| 13 | `S2_B3` | Sentinel-2 green |
| 14 | `S2_B4` | Sentinel-2 red |
| 15 | `S2_B8` | Sentinel-2 NIR |
| 16 | `S2_B11` | Sentinel-2 SWIR |
| 17 | `AirTemp_2m_C` | |
| 18 | `Dewpoint_2m_C` | |

---

## Spatial split

The spatial partition is defined over the BBMP modeling area (a user-supplied
boundary GeoJSON). The current implementation is a **directional-quantile**
partition (north-up):

```
TEST        = north-most 20% of the area
VALIDATION  = east-most 15% of the remaining area
TRAIN       = everything else (central / south / west)
```

Current pixel-level proportions:

```
TRAIN       64.92%
VALIDATION  15.02%
TEST        20.07%
```

**Important caveats**

- This split is **artificial and directional**; it is **NOT** an official
  administrative partition of Bengaluru.
- It exists to **reduce spatial leakage** (neighbouring pixels are highly
  correlated).
- Patches must satisfy a **≥ 90% purity** rule; patches crossing split
  boundaries are **rejected**, not reassigned.
- Because of the purity rule, **accepted-sample** proportions differ from
  pixel proportions:

  ```
  TRAIN       ~72.9%
  VALIDATION  ~11.0%
  TEST        ~16.1%
  ```

  This drift is expected and reported, not "fixed".

---

## Patch configuration

```
PATCH_SIZE = 128
STRIDE     = 64
PURITY     >= 0.90
```

A pixel belongs to exactly one split. A 128×128 candidate patch is accepted
only if at least 90% of its pixels have the target split's id (pixels outside
the modeling area count against purity). The purity constraint prevents patches
that straddle two regions from leaking information across the leakage barrier.

---

## Temporal formulation

```
12 consecutive historical months  →  next month's LST
```

Examples:

```
2019-01 ... 2019-12  →  2020-01
2019-02 ... 2020-01  →  2020-02
```

With 72 months and a 12-month window there are `72 − 12 = 60` temporal
prediction positions per accepted patch. With the current 118 accepted patches:
`118 × 60 = 7080` projected samples. Temporal order is never shuffled during
preprocessing.

---

## Missing data and normalization (TRAIN-only)

Order of operations is strict and uses **training data only**:

```
identify valid TRAIN pixels
        ↓
compute TRAIN-only per-channel fill value  (= TRAIN channel mean)
        ↓
fill NaN/Inf with those fill values  (applied to train, val and test alike)
        ↓
compute TRAIN-only per-channel normalization mean/std
        ↓
normalize with TRAIN statistics
```

- Fill values and normalization statistics are computed from **TRAIN pixels
  only**. Validation and test data never influence them.
- Each input channel is z-scored: `(x − train_mean) / train_std`.
- The target LST is normalized using the **same** LST mean/std as the historical
  LST input channel.
- Zero/near-zero std is guarded (std set to 1) so no NaN/Inf is produced.

### Target validity mask

The target LST validity mask is captured **before** filling:

```
1 = original target pixel was valid
0 = original target pixel was missing/invalid
```

Invalid target pixels are filled with the TRAIN LST mean as a placeholder. The
filled value **must not be scored as a real observation**; the mask exists so
the model can apply a masked loss.

---

## Terrain handling

`Elevation` and `Slope` are treated as static/near-static terrain information
while the 18-channel representation is retained. Each terrain channel is
replaced by its **per-pixel temporal mean over the run's months** (`nanmean`),
applied identically to every timestep. For the lazy dataset this mean is
**precomputed once** into `terrain_mean.tif` and read as a 128×128 window per
sample (an exact reproduction of the per-run temporal mean, not a redesign).

---

## Lazy dataset architecture

Materializing the full dataset as `.pt` files would require roughly **101 GB**
of duplicated data. Instead, the canonical monthly TIFFs remain the data source
and the dataset reconstructs samples **on demand in memory**:

```
processed TIFFs
      ↓
sample_index.jsonl   patch_index.json   month_catalog.json
normalization_stats.json   terrain_mean.tif
      ↓
UHIDataset
      ↓
windowed RasterIO reads (18 × 128 × 128 per month)
      ↓
fill + normalize
      ↓
[12, 18, 128, 128]
      ↓
DataLoader
      ↓
[B, 12, 18, 128, 128]
```

The authoritative implementation is **`uhi_split/dataset.py`**. When the
pipeline runs it copies that exact file into the output dataset directory as
`dataset.py`, so there is a single source of truth.

### Expected tensor shapes

```
Sample:      X [12, 18, 128, 128]   y [1, 128, 128]   mask [1, 128, 128]
Batched:     X [B, 12, 18, 128, 128]  y [B, 1, 128, 128]  mask [B, 1, 128, 128]
dtype:       X, y = float32;  mask = bool
```

---

## Usage

### 1. Configure paths

Edit `uhi_split/preprocess_config.yaml` and set your own paths (they are
placeholders by default):

```yaml
paths:
  input_dir: /path/to/processed_temporal   # monthly *18features*.tif files
  output_dir: /path/to/processed_dataset   # lightweight artifact output
  bbmp_boundary: /path/to/boundary.geojson # polygon of the modeling area
```

The loader validates that `input_dir` and `bbmp_boundary` exist and fails loudly
otherwise.

### 2. Build the dataset artifact (no tensors are written)

```bash
# Small, fast run over the earliest months (validates the whole pipeline):
python preprocess.py \
    --input  /path/to/processed_temporal \
    --output /tmp/uhi_test_dataset \
    --mode test

# Full run over all 72 months: builds the complete index and final
# TRAIN-only statistics. Still writes NO tensors.
python preprocess.py \
    --input  /path/to/processed_temporal \
    --output /path/to/processed_dataset \
    --mode full
```

Output artifact (lightweight):

```
processed_dataset/
├── sample_index.jsonl        # one deterministic record per sample
├── patch_index.json          # accepted patches + purity
├── month_catalog.json        # explicit month → file mapping
├── normalization_stats.json  # fill values + train means/stds
├── split_config.json         # split logic/config + percentages
├── metadata.json             # environment, projections, tensor shapes
├── terrain_mean.tif          # precomputed static-terrain temporal mean
├── dataset.py                # authoritative UHIDataset (copied verbatim)
├── README.md                 # full project context (PROJECT_README.md)
├── _effective_config.yaml    # exact config used
└── diagnostics/
```

### 3. Verify

```bash
python verify_dataset.py --dataset-root /path/to/processed_dataset
```

### 4. Use the Dataset / DataLoader

```python
from dataset import UHIDataset            # copied into processed_dataset/
from torch.utils.data import DataLoader

train_ds = UHIDataset("processed_dataset", split="train")
val_ds   = UHIDataset("processed_dataset", split="val")
test_ds  = UHIDataset("processed_dataset", split="test")

X, y, mask = train_ds[0]
print(X.shape, y.shape, mask.shape)       # [12,18,128,128] [1,128,128] [1,128,128]

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
Xb, yb, mb = next(iter(train_loader))
print(Xb.shape, yb.shape, mb.shape)       # [8,12,18,128,128] [8,1,128,128] [8,1,128,128]
```

For a masked loss, use `mb` to ignore invalid target pixels; filled target
values are placeholders and must not be scored.

---

## Validation status (data-pipeline only)

The following were verified against the local (private) dataset. These are
preprocessing/data-loading checks, **not** model-performance claims:

- No NaN/Inf in generated tensors
- Correct temporal labels (`12 historical months → immediately following month`)
- Deterministic sample indexing (no reliance on filesystem ordering)
- DataLoader tested with `num_workers=0` and `num_workers=4`, `pin_memory=True`
- The lazy `UHIDataset` reproduces the earlier reference implementation to
  within **≤ 2.4e-6** (float32 precision)
- No spatial leakage: accepted patches have 0% cross-split pixels

---

## Reproducibility

Given the input TIFFs plus the saved `normalization_stats.json`,
`split_config.json`, `sample_index.jsonl`, `patch_index.json`,
`month_catalog.json` and `terrain_mean.tif`, `UHIDataset` always reconstructs
the same sample. There is no randomness: patches are enumerated
deterministically, samples are ordered by `sample_id`, months have explicit
indices, and fixed TRAIN-only statistics are reused.

To reproduce the dataset artifact:

```
input data + config (channel order, split, patch size, stride, purity,
sequence length) → preprocess.py → sample/patch/month index + statistics
```

---

## Known limitations / open decisions

These are deliberately left open and are **not** silently resolved:

1. **Artificial spatial split.** The directional partition is not an official
   administrative definition; it should be approved or replaced with
   ward-based polygons for administratively meaningful results.
2. **Sample-level split drift** (≈72.9 / 11.0 / 16.1 %) caused by the 90%
   purity rule; the 65/15/20 target refers to the spatial partition only.
   Reported, not corrected.
3. **Rare source-data outliers** in some dynamic channels (e.g. EVI/NDVI/MNDWI
   outside plausible ranges, < 0.006% of sampled pixels). These are not clipped.
4. **Static terrain treatment** (temporal mean, 18 channels retained).
5. **Lazy-loading throughput / I/O** depends on storage and `num_workers`;
   caching/chunking policy is a modelling-stage decision.

---

## Requirements

```
pip install -r requirements.txt
```

`numpy`, `rasterio`, `affine`, `PyYAML`, `torch`, `geopandas`, `shapely`,
`matplotlib`.

---

## Next stage (separate repository / work)

```
UHIDataset → DataLoader → ConvLSTM → LST(t+1)
    → RMSE / MAE / R² → actual vs predicted spatial maps
    → (later) building-footprint counterfactual scenario engine
```

The later building-footprint component is a separate scenario/counterfactual
layer:

```
baseline input            → LST_baseline
modified building-footprint input → LST_scenario
ΔLST = LST_scenario − LST_baseline
```

This is a **model-based scenario estimate**, **not** direct causal attribution.

---

## Repository layout

```
UHI-PreProcessing/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── preprocess.py              # CLI entry point (test / full)
├── verify_dataset.py          # standalone dataset verifier
└── uhi_split/
    ├── __init__.py
    ├── preprocess_config.yaml # configuration (paths are placeholders)
    ├── config.py              # config load + validation
    ├── months.py              # discovery, chronology, grid validation
    ├── split.py               # spatial partition + masks
    ├── patches.py             # patch enumeration + purity rule
    ├── stats.py               # TRAIN-only fill + normalization statistics
    ├── terrain.py             # static-terrain temporal-mean precompute
    ├── index.py               # deterministic sample/patch/month index
    ├── dataset.py             # authoritative UHIDataset (lazy)
    ├── viz.py                 # diagnostic plots
    ├── pipeline.py            # orchestration + lazy validation
    ├── PROJECT_README.md      # full project/scientific context
    └── README.md
```

> **Data are not included.** The monthly GeoTIFFs, the generated dataset
> artifact, and the BBMP boundary GeoJSON are supplied by the user and are
> excluded from version control (see `.gitignore`). This repository contains
> code, a configuration template, and documentation only.

`uhi_split/PROJECT_README.md` contains the fuller scientific context
(dataset, channel semantics, validated results, research-paper notes).