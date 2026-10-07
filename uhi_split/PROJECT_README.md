# Bengaluru Urban Heat Island (UHI) Forecasting — Project Context

> **Purpose of this document.** This is persistent project context, not just
> software documentation. It records the scientific and data-processing
> decisions made so far so that the ConvLSTM modelling stage and later the
> research paper can be built on a stable, explicit foundation.
>
> **Status legend used throughout:**
> - **[VALIDATED]** — verified against the data/on disk in this repository.
> - **[DECISION]** — a project choice that was made deliberately.
> - **[ASSUMPTION]** — a working assumption that may need revisiting.
> - **[OPEN]** — unresolved; requires human approval or further work.

---

## 1. Project overview

The project forecasts **next-month land surface temperature (LST)** for the
Bengaluru urban area from a history of satellite/derived spatial features, using
a spatiotemporal model (ConvLSTM). The eventual goal includes a **building-
footprint counterfactual scenario engine** (see §12), but that stage is out of
scope for the current preprocessing work.

Model formulation (target):

```text
Input :  X = [B, 12, 18, 128, 128]   (12 past months, 18 feature channels)
Target:  y = [B,  1, 128, 128]       (LST, next month)
Mask  :  m = [B,  1, 128, 128]       (1 = valid target pixel, 0 = missing/filled)
```

---

## 2. Dataset

**[VALIDATED]** Inspected directly on disk
(the `processed_temporal/` input directory supplied via `paths.input_dir`):

| Property | Value |
|---|---|
| Files | 72 monthly GeoTIFFs, one per month, complete Jan 2019 → Dec 2024 |
| Filename pattern | `Bengaluru_YYYY_MM_18features.tif` (organized in yearly subdirs) |
| Grid | 1608 (W) × 1493 (H) pixels |
| CRS | EPSG:32643 (UTM zone 43N) |
| Pixel size | 30 m (native, **not resampled**) |
| Transform | origin (757950, 1458780), pixel 30 m, north-up |
| Bands | 18, `float32`, LZW-compressed, tiled |
| NoData | NaN (per band) |
| Missingness | ~2.0–2.8% overall; **no Inf**; lower inside the BBMP area (~0–1.9%) |
| Band order | Exactly matches the 18-channel list (verified from band descriptions) |

**[VALIDATED]** All 72 files share identical dimensions, transform, CRS, dtype
and band descriptions. Loading fails loudly if any file disagrees.

### Preprocessing already performed **[DECISION]**

The previous pipeline (left untouched by this stage) produced these files:

- Per-month feature computation from raw sources (Landsat ST, Sentinel-2,
  derived indices, fractions, terrain, atmospheric variables — see §3).
- **[VALIDATED]** **Temporal interpolation** across months: per-pixel **linear**
  interpolation along the time axis, filling NaN only, never altering valid
  values, never extrapolating leading/trailing gaps, gaps > 12 months left as
  NaN. No spatial interpolation.
- This stage does **not** rerun interpolation and does **not** modify the TIFFs.

Data sources (from the existing config, not re-derived here): Landsat Collection
2 Level-2 surface temperature; Sentinel-2 surface reflectance; SRTM terrain;
WorldCover (built-up mask, 10 m 2021); BBMP administrative boundary. **Source
details for every channel beyond what the repository records are not restated
here** — do not invent them.

---

## 3. Feature / channel table (18 channels, exact tensor order)

The order below is authoritative and is checked against the TIFF band
descriptions at load time.

| Idx | Name | Meaning | Temporal status | Normalization |
|----:|---|---|---|---|
| 0 | `LST_Celsius` | Land surface temperature (°C) — also the prediction target | dynamic | z-score (train) |
| 1 | `NDVI` | Normalized Difference Vegetation Index | dynamic | z-score (train) |
| 2 | `NDBI` | Normalized Difference Built-up Index | dynamic | z-score (train) |
| 3 | `MNDWI` | Modified Normalized Difference Water Index | dynamic | z-score (train) |
| 4 | `EVI` | Enhanced Vegetation Index | dynamic | z-score (train) |
| 5 | `Albedo` | Surface albedo | dynamic | z-score (train) |
| 6 | `Vegetation_Frac` | Vegetation fraction (land-cover) | dynamic | z-score (train) |
| 7 | `Impervious_Frac` | Impervious/built fraction | dynamic | z-score (train) |
| 8 | `Soil_Frac` | Soil fraction | dynamic | z-score (train) |
| 9 | `Elevation` | Elevation (m) | **static** | z-score (train, temporal-mean) |
| 10 | `Slope` | Slope (deg) | **static** | z-score (train, temporal-mean) |
| 11 | `S2_B2` | Sentinel-2 band 2 (blue) reflectance | dynamic | z-score (train) |
| 12 | `S2_B3` | Sentinel-2 band 3 (green) reflectance | dynamic | z-score (train) |
| 13 | `S2_B4` | Sentinel-2 band 4 (red) reflectance | dynamic | z-score (train) |
| 14 | `S2_B8` | Sentinel-2 band 8 (NIR) reflectance | dynamic | z-score (train) |
| 15 | `S2_B11` | Sentinel-2 band 11 (SWIR) reflectance | dynamic | z-score (train) |
| 16 | `AirTemp_2m_C` | 2 m air temperature (°C) | dynamic | z-score (train) |
| 17 | `Dewpoint_2m_C` | 2 m dewpoint (°C) | dynamic | z-score (train) |

**[VALIDATED]** `Elevation` and `Slope` are (near-)static in time. `Slope` is
*exactly* identical across all months. `Elevation` is *almost* static: a small
number of pixels can differ by up to **1 m** between months (a rounding artefact
of the source), and its NaN pattern can vary slightly between months.

**[DECISION]** Terrain treatment (unchanged from the validated implementation):
each terrain channel is replaced by its **per-pixel temporal mean over the run's
months** (`nanmean`), and this same value is used for every timestep. In the
lazy architecture this mean is **precomputed once** into `terrain_mean.tif`
(bands `Elevation`, `Slope`) and read as a 128×128 window per sample — an exact
reproduction of the previous per-run `nanmean`, not a redesign.

---

## 4. Temporal formulation

**[DECISION]**

```text
X = months [t-11 ... t]   (12 consecutive monthly feature stacks)
y = LST at month [t+1]
```

Example:

```text
X: 2019-01, 2019-02, ..., 2019-12
y: 2020-01

X: 2019-02, 2019-03, ..., 2020-01
y: 2020-02
```

**[VALIDATED]** With 72 months and a 12-month input window plus a 1-month
target, there are `72 - 12 = 60` temporal prediction positions per patch. Total
projected full-run samples: `118 patches × 60 = 7080`. Temporal order is never
shuffled during preprocessing (the `sample_index` is chronological).

---

## 5. Spatial split

**[VALIDATED]** Boundary: supplied via `paths.bbmp_boundary`
— 243 wards, EPSG:4326, 6 invalid geometries repaired with `make_valid`, union
area **717.1 km²**, fully contained inside the raster footprint. The modeling
area is the **BBMP union** (BBMP covers only ~33% of the raster rectangle).

### Construction of the candidate split **[DECISION]**

A **directional-quantile** partition of the BBMP area (north-up, `rotation=0`):

1. **TEST = north-most 20%** by northing.
2. **VAL = east-most 15%** of the remaining area.
3. **TRAIN = everything else** (central/south/west).
4. Optional `assignment_buffer_m` can insert a train-only gap at the test
   boundary; default `0`.

**[VALIDATED]** Each split is a **single contiguous connected region** (100% of
its pixels in one 4-connected component).

| Split | Intended meaning | Pixel % **[VALIDATED]** | Accepted patches **[VALIDATED]** |
|---|---|---|---|
| train | Central + South | 64.92% | 86 |
| val | East | 15.02% | 13 |
| test | North + buffer | 20.07% | 19 |

### Important caveats

- **[OPEN] The split is artificial and directional.** It is **NOT** an official
  administrative partition of Bengaluru. `split_config.json` records
  `"is_official_admin_boundary": false`. Human approval or replacement with
  ward-based polygons is required before treating results as administratively
  meaningful.
- **[VALIDATED] Sample-level proportions drift from pixel-level** because of the
  purity rule: projected **train 72.9% / val 11.0% / test 16.1%**. This is a
  consequence of the 90% purity threshold, **not** a bug, and it must be
  **reported rather than silently corrected** (see §10, open decision 2).

---

## 6. Patch generation and purity

**[DECISION]**

```text
PATCH_SIZE = 128
STRIDE     = 64
PURITY     >= 0.90
```

- Candidate windows are enumerated across the raster with the configured stride.
- A pixel has exactly one split id (train/val/test, or "outside").
- A patch is accepted into a split only if **≥ 90%** of its pixels have that
  split's id. Pixels outside the BBMP area count against purity.
- If no split reaches 90%, the patch is **rejected** and never reassigned.
- Because pixel ids are exclusive, a patch can never enter more than one split.

**[VALIDATED]** 575 candidates → **118 accepted** (79.5% rejected, all with
best-split purity < 0.90; max 0.8929). **Zero cross-split pixel
contamination** among accepted patches (cross-split fractions are 0.00%;
only outside-BBMP pixels reduce purity). Minimum accepted purity 0.9003.

Why reject boundary-crossing patches: a 128×128 window straddling two regions
would mix pixels that the split is specifically designed to keep apart, creating
spatial leakage (neighbouring pixels are highly correlated). Rejecting them
enforces the leakage barrier at the cost of losing boundary patches.

---

## 7. Missing-data handling

**[DECISION]** Order of operations (strict; no validation/test influence):

```text
1. identify valid TRAIN pixels
2. compute TRAIN-only per-channel fill value (= TRAIN channel mean)
3. fill NaN/Inf with those fill values (train, val, test alike)
4. compute TRAIN-only per-channel normalization mean/std
5. normalize with TRAIN statistics
```

- **[DECISION]** The fill statistic is the **TRAIN channel mean** (one value per
  channel), applied identically to all splits. Val/test never contribute.
- **[VALIDATED]** `normalization_stats.json` stores `fill_values`,
  `train_means`, `train_stds`, `valid_pixel_counts`, `gap_fraction`,
  `degenerate_channels`, `over_gap_channels`, `min_std_guard`.

### Target LST and validity mask **[DECISION]**

- The target validity mask is captured **before** filling:
  `1 = original target pixel valid`, `0 = missing/filled`.
- Invalid target pixels are filled with the TRAIN LST mean.
- **[VALIDATED]** The filled target value equals the LST TRAIN mean and is
  **not** treated as an observation. Normalized filled targets are exactly `0`
  (only used as a placeholder); the mask exists so the future model can apply a
  **masked loss**. Filled values must never be scored as genuine observations.
- **[DECISION]** The target LST is normalized with the **same** LST mean/std as
  the historical LST input channel.

---

## 8. Normalization

**[DECISION]** Per-channel z-score using **TRAIN-only** statistics:

```text
x_normalized = (x - train_mean) / train_std
```

- Applies to all 18 input channels.
- Target LST uses the LST channel's mean/std.
- Zero/near-zero std is guarded: `std := 1.0` and the channel is recorded in
  `degenerate_channels` (never produce NaN/Inf).
- **[VALIDATED]** No NaN/Inf in the produced samples; statistics are stored in
  `normalization_stats.json`.

---

## 9. Lazy dataset architecture

**[DECISION]** The full X/y/mask dataset is **not materialized**. Storing all
7080 samples would be ≈101 GB of duplicated data derived from the ~9–10 GB of
TIFFs. Instead, the canonical monthly TIFFs remain the data source and a
lightweight index + `UHIDataset` reconstruct samples on demand:

```text
processed_temporal/ TIFFs (canonical, ~9–10 GB)
        ↓
sample_index.jsonl + patch_index.json + month_catalog.json + stats
        ↓
UHIDataset.__getitem__
        ↓
windowed rasterio reads (only 18×128×128 per month)
        ↓
terrain_mean window + fill + normalize
        ↓
[12,18,128,128]  /  [1,128,128]  /  [1,128,128]
        ↓
DataLoader
        ↓
[B,12,18,128,128]
        ↓
ConvLSTM
```

**[DECISION]** There is exactly **one authoritative implementation**, the source
file `uhi_split/dataset.py`. The preprocessing pipeline **copies it verbatim**
into the output directory as `dataset.py`, so the deployed copy is never a
second, independently maintained codebase.

**[VALIDATED]** Windowed reads only (`rasterio` windows); no 200–250 MB TIFF is
loaded per sample. A per-process reader cache is opened lazily after DataLoader
workers fork, so it is safe with `num_workers>0` / `persistent_workers=True`.
Verified with `num_workers` 0 and 4, `pin_memory=True`, batch shape
`[4,12,18,128,128]`, deterministic output, no NaN/Inf.

### Output artifact

```text
processed_dataset/
├── sample_index.jsonl        # one deterministic record per sample
├── patch_index.json          # accepted patches + purity
├── month_catalog.json        # explicit month index -> relative path
├── normalization_stats.json  # fill_values, train_means, train_stds, ...
├── split_config.json         # exact split logic/config + percentages
├── metadata.json             # environment, projections, tensor shapes
├── terrain_mean.tif          # precomputed static-terrain temporal mean
├── dataset.py                # authoritative UHIDataset (copied)
├── README.md                 # this document (copied)
├── _effective_config.yaml    # exact config used
└── diagnostics/
    ├── split_mask.tif  region_mask.tif
    ├── spatial_split_map.png  split_profiles.png  sample_lst_patch.png
    └── lazy_validation.json
```

### Sample record schema **[VALIDATED, deterministic]**

Each line of `sample_index.jsonl`:

```json
{
  "sample_id": 0,
  "split": "train",
  "patch_id": 0,
  "row_off": 576, "col_off": 384,
  "height": 128, "width": 128,
  "seq_start_index": 0,
  "input_month_indices": [0,1,...,11],
  "target_month_index": 12,
  "input_months": ["2019-01", ..., "2019-12"],
  "target_month": "2020-01"
}
```

Nothing relies on implicit filesystem ordering: months, patches and samples all
carry explicit integer ids and are re-sorted by `sample_id` on load.

---

## 10. Current validated results (test mode)

**[VALIDATED]** Test build over the first 24 months (12 temporal positions):

```text
train: n=1032  X[12,18,128,128]  y[1,128,128]  mask[1,128,128]
val  : n=156   X[12,18,128,128]  y[1,128,128]  mask[1,128,128]
test : n=228   X[12,18,128,128]  y[1,128,128]  mask[1,128,128]
NaN=0, Inf=0 (all splits)
```

- **[VALIDATED] Temporal verification:** samples are exactly
  `2019-01…2019-12 → 2020-01`, and consecutive positions shift by one month.
- **[VALIDATED] No leakage:** no patch appears in more than one split; accepted
  patches have 0.00% cross-split pixels.
- **[VALIDATED] Lazy vs materialized agreement:** the lazy `UHIDataset`
  reproduces a previously materialized test build to **≤ 2.4e-6** (float32
  precision) for X, y and mask. The only difference in statistics is terrain
  std (~1e-5, float64 summation order), scientifically negligible.
- **[VALIDATED] Mask semantics:** on a real target with missing pixels
  (2024-07), `mask == isfinite(raw_target)`, and filled targets normalize to 0.
- **[VALIDATED] Split:** pixel 64.92 / 15.02 / 20.07 (train/val/test), each
  contiguous.

---

## 11. Known concerns / open decisions

1. **[OPEN] Artificial spatial split.** The directional partition is not an
   official administrative definition. Approve it, adjust `rotation_deg` /
   `test_fraction` / `val_fraction` / `assignment_buffer_m`, or replace it with
   ward-based polygons before the full run.
2. **[OPEN] Sample-level split drift** (72.9 / 11.0 / 16.1 vs 65/15/20) caused
   by the 90% purity rule. The target 65/15/20 refers to the **spatial
   partition**; the sample drift is a consequence and should be **reported, not
   silently corrected**. Decide whether the drift is acceptable.
3. **[OPEN] Rare source-data outliers.** A very small fraction (< 0.006% of
   sampled pixels) of dynamic channels fall outside plausible ranges
   (e.g. EVI → 3.08, NDVI → 1.70, MNDWI → −2.39). These are **not** clipped
   (out of scope; clipping would alter data). Decide whether to keep, flag, or
   add a documented outlier policy later.
4. **[OPEN] Static terrain.** `Elevation` almost static (≤1 m monthly jitter),
   `Slope` exactly static. Current treatment = per-pixel temporal mean applied
   to every timestep, keeping 18 channels. Alternative (future) would be to drop
   them to 16 channels or feed them once; **do not redesign silently**.
5. **[OPEN] Storage/throughput.** Lazy loading trades disk for per-access I/O.
   Throughput depends on storage and `num_workers`. Chunking/caching policy is
   a modelling-stage decision.

---

## 12. Next project stage

```text
UHIDataset  →  DataLoader  →  ConvLSTM  →  LST(t+1)
    →  RMSE / MAE / R²  →  actual vs predicted spatial maps
    →  (later) building-footprint counterfactual scenario engine
```

### Counterfactual formulation (later; not implemented yet)

```text
Baseline input           → LST_baseline(t+1)
Modified building-footprint input → LST_scenario(t+1)
ΔLST = LST_scenario − LST_baseline
```

**[DECISION]** This is explicitly a **model-based counterfactual / scenario
estimate**, **not** direct causal attribution. Any paper must phrase it that
way.

### How to use the Dataset in training **[VALIDATED example]**

```python
from dataset import UHIDataset            # copied into processed_dataset/
from torch.utils.data import DataLoader

train_ds = UHIDataset("processed_dataset", split="train")
val_ds   = UHIDataset("processed_dataset", split="val")
test_ds  = UHIDataset("processed_dataset", split="test")

X, y, mask = train_ds[0]
print(X.shape, y.shape, mask.shape)       # torch.Size([12,18,128,128]) ...

loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,                # shuffling is fine for training; order is
                                 # deterministic per-sample regardless
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
Xb, yb, mb = next(iter(loader))
print(Xb.shape, yb.shape, mb.shape)       # [8,12,18,128,128] / [8,1,128,128] / ...
```

For a masked loss, use `mb` to zero-out (or ignore) invalid target pixels;
filled target values are placeholders and must not be scored.

---

## 13. Research-paper notes (methodological facts to reflect later)

- **Temporal coverage:** January 2019 – December 2024 (72 monthly composites).
- **Spatial resolution / CRS:** 30 m, EPSG:32643; extent 1608 × 1493.
- **Feature set:** 18 channels (see §3); LST is both feature and target;
  Elevation/Slope are static terrain.
- **Interpolation:** per-pixel temporal linear interpolation (previous stage),
  NaN-only, no extrapolation, ±12-month gap limit, no spatial interpolation.
- **Spatial split:** artificial directional-quantile partition of the BBMP area
  (train 64.92% / val 15.02% / test 20.07% by pixels), each contiguous; **not**
  administrative. Boundary-crossing patches rejected at 90% purity.
- **Normalization:** train-only per-channel z-score; fill values = train means;
  target normalized with the LST input statistics.
- **Temporal formulation:** 12-month history → next-month LST; 60 positions.
- **Patch extraction:** 128×128, stride 64, ≥90% purity, no cross-split patches.
- **Missing-data treatment:** temporal interpolation + train-mean fill; target
  validity mask retained for a masked loss.
- **ConvLSTM formulation:** *to be defined in the modelling stage* (not yet
  established — do not state architecture claims that haven't been computed).
- **Evaluation plan:** RMSE / MAE / R² on the spatially held-out test region,
  plus actual-vs-predicted spatial maps. *Numbers not yet produced.*
- **Limitations (to expand):** single-city scope; artificial split; static
  terrain; residual source outliers; missingness in the target; the split/leakage
  trade-off; counterfactual results are model-based scenarios, not causal
  effects.
- **Counterfactual/scenario methodology:** baseline vs modified building-
  footprint inputs → ΔLST; model-based scenario estimate, not causal attribution.

**[OPEN] No citations, scientific claims, or results are asserted here beyond
what has actually been computed and verified.** The paper must not fabricate
citations or results.

---

## 14. Reproducibility summary

Given `processed_temporal/`, `normalization_stats.json`, `split_config.json`,
`sample_index.jsonl`, `patch_index.json`, `month_catalog.json` and
`terrain_mean.tif`, `UHIDataset` reconstructs exactly the same sample every
time. All randomness is absent: patches are enumerated deterministically,
samples are ordered by `sample_id`, months by explicit index, and the same
train-only statistics are reused unchanged. Random splitting of pixels or
patches is never performed.