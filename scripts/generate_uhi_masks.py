#!/usr/bin/env python3
"""Generate full-study-area urban and rural reference masks for SUHI analysis.

This is an ADDITIVE utility. It does not touch the frozen preprocessing
pipeline, the dataset contract, channel ordering, normalization, patch
generation or the train/val/test split. It only reads canonical inputs and
writes new mask rasters.

What it produces
----------------
After ConvLSTM inference the project derives Surface Urban Heat Island (SUHI)
intensity from a predicted LST field as::

    SUHI = mean(LST_urban) - mean(LST_rural)

The ``urban_mask`` and ``rural_mask`` produced here are full-study-area,
pixel-for-pixel aligned to the 30 m model grid. They are intersected later
(during inference/evaluation) with the predicted LST grid / 128x128 patches so
that the urban/rural definition is globally consistent across all predictions.

Urban mask
----------
``WorldCover == 50`` (Built-up), clipped to the study boundary.

Rural reference mask
--------------------
A 5 km outward peri-urban ring around the urban extent, minus the urban extent,
filtered to vegetated / open WorldCover classes and clipped to the study
boundary::

    rural_candidate = 5 km outward buffer(urban extent)  -  urban extent
    rural_mask      = rural_candidate
                      & WorldCover in {10, 20, 30, 40, 60}
                      & study boundary

Retained classes : 10 Tree cover, 20 Shrubland, 30 Grassland,
                   40 Cropland, 60 Bare / sparse vegetation
Excluded classes : 50 Built-up, 70 Snow and ice, 80 Permanent water bodies,
                   90 Herbaceous wetland, 95 Mangroves, 100 Moss and lichen,
                   plus 0 (WorldCover nodata / outside tile coverage)

The class legend is read from the WorldCover GeoTIFF's ``legend`` tag and
validated; hard-coded assumptions are only a fallback (a warning is printed).

Grid contract
-------------
The masks are written on the EXACT grid of ``--reference-grid`` (CRS, transform,
pixel size, width, height, extent). The 10 m WorldCover raster is reprojected to
that grid with a CATEGORICAL ``mode`` (majority) aggregation, matching the
existing preprocessing contract (``spatial/urban_mask.py``), and is never used
directly as the output resolution.

Usage
-----
    python scripts/generate_uhi_masks.py \\
        --worldcover /path/to/ESA_WorldCover_10m_2021_v200_N12E075_Map.tif \\
        --study-boundary /path/to/BBMP.geojson \\
        --reference-grid /path/to/monthly_18features.tif \\
        --output-dir /path/to/uhi_masks
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy import ndimage

# --- Frozen methodology constants -------------------------------------------
URBAN_CLASS = 50
RURAL_CLASSES = (10, 20, 30, 40, 60)
EXCLUDED_CLASSES = (50, 70, 80, 90, 95, 100)
RURAL_BUFFER_M = 5000.0
MASK_NODATA = 0
MASK_DTYPE = "uint8"

# Fallback legend for ESA WorldCover v200 (used only if the GeoTIFF tag is
# absent; a warning is emitted so the operator can confirm the product version).
WORLDCOVER_FALLBACK_LEGEND = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


@dataclass
class ReferenceGrid:
    path: str
    crs: str
    transform: tuple
    width: int
    height: int
    res: tuple
    nodata: object
    dtype: str


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def read_reference_grid(path: str | Path) -> ReferenceGrid:
    """Snapshot the authoritative model grid. The grid is never resampled."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference grid not found: {path}")
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError(f"Reference grid '{path}' has no CRS; refusing to guess.")
        return ReferenceGrid(
            path=str(path),
            crs=ds.crs.to_string(),
            transform=tuple(ds.transform)[:6],
            width=ds.width,
            height=ds.height,
            res=(abs(ds.transform.a), abs(ds.transform.e)),
            nodata=ds.nodata,
            dtype=ds.dtypes[0],
        )


def read_worldcover_legend(path: str | Path) -> tuple[dict[int, str], str, str]:
    """Parse the WorldCover class legend from the GeoTIFF metadata tags.

    Returns (legend, product_version, source) where source is 'tag' or 'fallback'.
    """
    with rasterio.open(path) as ds:
        tags = ds.tags()
    version = tags.get("algorithm_version") or tags.get("product_version") or "unknown"
    raw = tags.get("legend")
    if not raw:
        print(
            "  [warn] WorldCover GeoTIFF has no 'legend' tag; using the built-in "
            "ESA WorldCover v200 fallback legend.",
            file=sys.stderr,
        )
        return dict(WORLDCOVER_FALLBACK_LEGEND), version, "fallback"
    legend: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        legend[int(parts[0])] = parts[1].strip()
    if not legend:
        print(
            "  [warn] Could not parse WorldCover 'legend' tag; using fallback.",
            file=sys.stderr,
        )
        return dict(WORLDCOVER_FALLBACK_LEGEND), version, "fallback"
    return legend, version, "tag"


def align_worldcover(
    worldcover_path: str | Path, grid: ReferenceGrid
) -> np.ndarray:
    """Reproject 10 m WorldCover onto the model grid using categorical mode.

    Mirrors the frozen preprocessing contract: categorical majority aggregation,
    uint8, src/dst nodata = 0. The 10 m raster is never used as final output.
    """
    from affine import Affine

    transform = Affine(*grid.transform)
    with rasterio.open(worldcover_path) as src:
        if src.crs is None:
            raise ValueError("WorldCover raster has no CRS; refusing to guess.")
        dst = np.zeros((grid.height, grid.width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else 0,
            dst_transform=transform,
            dst_crs=grid.crs,
            dst_nodata=0,
            resampling=Resampling.mode,
        )
    return dst


def rasterize_study_boundary(boundary_path: str | Path, grid: ReferenceGrid) -> np.ndarray:
    """Rasterize the study boundary onto the model grid (all_touched=False)."""
    import geopandas as gpd
    from shapely import make_valid

    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Study boundary not found: {boundary_path}")
    gdf = gpd.read_file(boundary_path)
    if gdf.empty:
        raise ValueError(f"Study boundary is empty: {boundary_path}")
    if gdf.crs is None:
        raise ValueError("Study boundary has no CRS; refusing to guess.")
    if gdf.crs.to_string() != grid.crs:
        gdf = gdf.to_crs(grid.crs)
    if not bool(gdf.geometry.is_valid.all()):
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    footprint = gdf.geometry.union_all()
    from affine import Affine

    return rasterize(
        [(footprint, 1)],
        out_shape=(grid.height, grid.width),
        transform=Affine(*grid.transform),
        fill=0,
        all_touched=False,
        dtype=np.uint8,
    )


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------
def outward_buffer_ring(
    urban: np.ndarray, buffer_m: float, pixel_m: float
) -> np.ndarray:
    """Pixels strictly outside ``urban`` but within ``buffer_m`` of it.

    Uses a Euclidean distance transform on the projected raster grid (metres),
    so the ring is a true outward buffer minus the urban extent. The urban grid
    is in a projected CRS, so planar distances are valid.
    """
    if not urban.any():
        raise ValueError("Urban extent is empty; cannot build a buffer around it.")
    # distance_transform_edt(~urban) -> for each non-urban pixel, the distance to
    # the nearest urban pixel, in pixel units; scale by pixel size to get metres.
    dist_px = ndimage.distance_transform_edt(~urban.astype(bool))
    dist_m = dist_px * float(pixel_m)
    ring = (dist_m > 0.0) & (dist_m <= float(buffer_m))
    return ring


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_binary_mask(
    path: Path,
    data: np.ndarray,
    grid: ReferenceGrid,
    *,
    description: str,
    overwrite: bool,
    extra_tags: dict | None = None,
) -> None:
    from affine import Affine

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing '{path}'. Pass --overwrite to allow it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "count": 1,
        "height": grid.height,
        "width": grid.width,
        "dtype": MASK_DTYPE,
        "crs": grid.crs,
        "transform": Affine(*grid.transform),
        "nodata": MASK_NODATA,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(MASK_DTYPE, copy=False), 1)
        dst.set_band_description(1, description)
        if extra_tags:
            dst.update_tags(**{k: str(v) for k, v in extra_tags.items()})


def save_validation_png(
    path: Path,
    urban: np.ndarray,
    rural: np.ndarray,
    study: np.ndarray,
) -> None:
    """Quick validation figure: urban / rural / other study / outside study."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # 0 = outside study (background), 1 = study non-urban non-rural, 2 = rural, 3 = urban
    img = np.zeros(urban.shape, dtype=np.uint8)
    img[study.astype(bool)] = 1
    img[rural.astype(bool)] = 2
    img[urban.astype(bool)] = 3
    # urban wins over everything, rural over study-other
    cmap = ListedColormap(["#f0f0f0", "#d9d9d9", "#2ca02c", "#d62728"])
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(img, cmap=cmap, interpolation="nearest", vmin=0, vmax=3)
    ax.set_title("UHI masks (validation only) — red=urban, green=rural")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="urban"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="rural reference"),
        plt.Rectangle((0, 0), 1, 1, color="#d9d9d9", label="study, other"),
        plt.Rectangle((0, 0), 1, 1, color="#f0f0f0", label="outside study"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_masks(
    urban: np.ndarray,
    rural: np.ndarray,
    worldcover: np.ndarray,
    known_classes: set[int],
    *,
    grid: ReferenceGrid,
    ref_meta: ReferenceGrid,
    rural_buffer_m: float,
    npixels_study: int,
    clip_to_boundary: bool = True,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    # Percentages are relative to the study area when clipped, otherwise to the
    # full reference grid the masks cover.
    denominator = npixels_study if clip_to_boundary else int(urban.size)

    if urban.shape != rural.shape:
        errors.append(f"shape mismatch: urban {urban.shape} vs rural {rural.shape}")
    if urban.shape != (ref_meta.height, ref_meta.width):
        errors.append(
            f"mask shape {urban.shape} != reference grid "
            f"({ref_meta.height}, {ref_meta.width})"
        )

    overlap = int(np.logical_and(urban.astype(bool), rural.astype(bool)).sum())
    if overlap != 0:
        errors.append(f"urban and rural masks overlap on {overlap} pixels (must be 0)")

    urban_n = int(urban.sum())
    rural_n = int(rural.sum())
    if urban_n == 0:
        errors.append("urban mask contains zero pixels")
    if rural_n == 0:
        errors.append("rural mask contains zero pixels")

    for name, arr in (("urban", urban), ("rural", rural)):
        uniq = set(np.unique(arr).tolist())
        if not uniq <= {0, 1}:
            errors.append(f"{name} mask has unexpected values: {sorted(uniq)}")

    wc_uniq = set(int(v) for v in np.unique(worldcover).tolist())
    unexpected = sorted(wc_uniq - known_classes - {0})
    if unexpected:
        warnings.append(
            f"aligned WorldCover contains class codes absent from the legend: {unexpected}"
        )

    if urban_n and rural_n:
        ratio = urban_n / rural_n
        if ratio > 50 or ratio < 1 / 50:
            warnings.append(
                f"urban/rural pixel ratio is extreme ({ratio:.3g}); "
                "check the buffer and study-boundary clip."
            )

    report = {
        "errors": errors,
        "warnings": warnings,
        "urban_pixels": urban_n,
        "rural_pixels": rural_n,
        "study_pixels": int(npixels_study),
        "overlap_pixels": overlap,
        "urban_pct_of_study": (100.0 * urban_n / denominator) if denominator else 0.0,
        "rural_pct_of_study": (100.0 * rural_n / denominator) if denominator else 0.0,
    }
    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--worldcover", required=True,
                   help="ESA WorldCover GeoTIFF (e.g. ESA_WorldCover_10m_2021_v200_N12E075_Map.tif)")
    p.add_argument("--study-boundary", required=True,
                   help="Study/modeling boundary polygon (GeoJSON/GPKG/...)")
    p.add_argument("--reference-grid", required=True,
                   help="Existing model-grid raster; defines CRS/transform/size/extent")
    p.add_argument("--output-dir", required=True, help="Where masks + metadata are written")
    p.add_argument("--rural-buffer-km", type=float, default=RURAL_BUFFER_M / 1000.0,
                   help="Outward buffer around the urban extent (km), default 5")
    p.add_argument("--urban-class", type=int, default=URBAN_CLASS,
                   help="WorldCover class treated as urban/built-up, default 50")
    p.add_argument("--no-clip-to-boundary", dest="clip_to_boundary",
                   action="store_false", default=True,
                   help="Do NOT clip masks to the study boundary (default: clip)")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting existing output files")
    p.add_argument("--no-viz", dest="viz", action="store_false", default=True,
                   help="Skip the validation PNG")
    return p


def run(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== UHI mask generation ===")
    print(f"worldcover     : {args.worldcover}")
    print(f"study-boundary : {args.study_boundary}")
    print(f"reference-grid : {args.reference_grid}")
    print(f"output-dir     : {out_dir}")

    # 1. Grid + legend
    grid = read_reference_grid(args.reference_grid)
    print(f"\n[grid] CRS {grid.crs} | size {grid.width} x {grid.height} | "
          f"res {grid.res[0]:g} m | transform {grid.transform}")
    legend, wc_version, legend_source = read_worldcover_legend(args.worldcover)
    print(f"[worldcover] version={wc_version} legend_source={legend_source}")

    # Validate class definitions rather than blindly assuming them.
    if grid.res[0] != 30.0 or grid.res[1] != 30.0:
        print(f"  [warn] reference grid resolution is {grid.res}, not the documented "
              "30 m; masks will match the grid you supplied.", file=sys.stderr)
    builtup_label = legend.get(args.urban_class, "")
    if "built" not in builtup_label.lower():
        print(
            f"  [warn] urban class {args.urban_class} label is '{builtup_label}' "
            "(expected a 'Built-up' label). Confirm the WorldCover version.",
            file=sys.stderr,
        )
    for c in RURAL_CLASSES:
        if c not in legend:
            print(f"  [warn] retained rural class {c} missing from legend.", file=sys.stderr)

    print("[classes] retained rural classes:")
    for c in RURAL_CLASSES:
        print(f"    {c:>3}  {legend.get(c, 'unknown')}")
    print("[classes] excluded classes:")
    for c in EXCLUDED_CLASSES:
        print(f"    {c:>3}  {legend.get(c, 'unknown')}")

    # 2. Align WorldCover to the model grid (categorical mode, 30 m)
    print("\n[align] WorldCover 10 m -> model grid (resampling=mode, categorical) ...")
    wc = align_worldcover(args.worldcover, grid)

    # 3. Study boundary on the model grid
    print("[study] rasterizing study boundary ...")
    study = rasterize_study_boundary(args.study_boundary, grid)
    n_study = int(study.sum())
    print(f"[study] study pixels: {n_study} ({100.0 * n_study / study.size:.2f}% of grid)")

    # 4. Urban mask = built-up within study
    urban_full = (wc == args.urban_class)
    urban = urban_full & study.astype(bool) if args.clip_to_boundary else urban_full.copy()

    # 5. Rural reference = 5 km outward buffer(urban) - urban, vegetated classes,
    #    clipped to study.
    print(f"[rural] building {args.rural_buffer_km:g} km outward peri-urban ring ...")
    ring = outward_buffer_ring(urban, args.rural_buffer_km * 1000.0, grid.res[0])
    rural_class_filter = np.isin(wc, np.asarray(RURAL_CLASSES, dtype=wc.dtype))
    rural = ring & rural_class_filter
    if args.clip_to_boundary:
        rural = rural & study.astype(bool)
    ring_pixels = int(ring.sum())
    dropped_by_class = int((ring & ~rural_class_filter).sum())

    urban = urban.astype(np.uint8)
    rural = rural.astype(np.uint8)

    # 6. Validation
    known = set(legend) | {0}
    report = validate_masks(
        urban, rural, wc, known,
        grid=grid, ref_meta=grid,
        rural_buffer_m=args.rural_buffer_km * 1000.0,
        npixels_study=n_study,
        clip_to_boundary=args.clip_to_boundary,
    )

    # 7. Write artifacts
    print(f"\n[write] {out_dir / 'urban_mask.tif'}")
    write_binary_mask(
        out_dir / "urban_mask.tif", urban, grid,
        description="Urban mask (WorldCover built-up within study boundary)",
        overwrite=args.overwrite,
        extra_tags={"worldcover_version": wc_version, "urban_class": args.urban_class},
    )
    print(f"[write] {out_dir / 'rural_mask.tif'}")
    write_binary_mask(
        out_dir / "rural_mask.tif", rural, grid,
        description="Rural reference mask (5 km peri-urban ring, vegetated classes)",
        overwrite=args.overwrite,
        extra_tags={
            "worldcover_version": wc_version,
            "rural_classes": ",".join(str(c) for c in RURAL_CLASSES),
            "buffer_m": int(args.rural_buffer_km * 1000),
        },
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "worldcover_version": wc_version,
        "worldcover_legend_source": legend_source,
        "worldcover_source": str(args.worldcover),
        "study_boundary_source": str(args.study_boundary),
        "reference_grid_source": str(args.reference_grid),
        "urban_class": args.urban_class,
        "urban_class_label": legend.get(args.urban_class, "unknown"),
        "rural_buffer_km": args.rural_buffer_km,
        "rural_buffer_m": args.rural_buffer_km * 1000.0,
        "rural_classes": list(RURAL_CLASSES),
        "rural_class_labels": {str(c): legend.get(c, "unknown") for c in RURAL_CLASSES},
        "excluded_classes": list(EXCLUDED_CLASSES),
        "excluded_class_labels": {str(c): legend.get(c, "unknown") for c in EXCLUDED_CLASSES},
        "clip_to_study_boundary": bool(args.clip_to_boundary),
        "crs": grid.crs,
        "resolution_m": grid.res[0],
        "width": grid.width,
        "height": grid.height,
        "transform": list(grid.transform),
        "reference_grid_nodata": grid.nodata,
        "mask_dtype": MASK_DTYPE,
        "mask_nodata": MASK_NODATA,
        "nodata_note": (
            "Mask nodata is 0, matching the existing frozen preprocessing mask "
            "artifacts (urban_mask_30m.tif, bbmp_mask.tif). Because the masks are "
            "binary, 0 simultaneously means 'false' and 'nodata'; valid data are "
            "the 0/1 values themselves."
        ),
        "urban_pixel_count": report["urban_pixels"],
        "rural_pixel_count": report["rural_pixels"],
        "urban_pct_of_study": report["urban_pct_of_study"],
        "rural_pct_of_study": report["rural_pct_of_study"],
        "percentage_basis": (
            "study_boundary" if args.clip_to_boundary else "full_reference_grid"
        ),
        "percentage_denominator_pixels": (
            report["study_pixels"] if args.clip_to_boundary
            else int(grid.width * grid.height)
        ),
        "study_pixel_count": report["study_pixels"],
        "urban_rural_overlap_pixels": report["overlap_pixels"],
        "rural_ring_pixel_count_before_class_filter": ring_pixels,
        "rural_ring_pixels_dropped_by_class_filter": dropped_by_class,
        "rural_method": (
            "5 km outward Euclidean distance-buffer around the urban extent, "
            "minus the urban extent, filtered to rural classes, clipped to study."
        ),
        "validation": {
            "errors": report["errors"],
            "warnings": report["warnings"],
        },
    }
    meta_path = out_dir / "mask_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[write] {meta_path}")

    if args.viz:
        png_path = out_dir / "mask_validation.png"
        save_validation_png(png_path, urban, rural, study)
        print(f"[write] {png_path}")

    # 8. Report
    basis = "study" if args.clip_to_boundary else "full grid"
    print("\n=== VALIDATION ===")
    print(f"urban pixels      : {report['urban_pixels']}")
    print(f"rural pixels      : {report['rural_pixels']}")
    print(f"study pixels      : {report['study_pixels']}")
    print(f"urban + rural     : "
          f"{report['urban_pixels'] + report['rural_pixels']} "
          f"({report['urban_pct_of_study'] + report['rural_pct_of_study']:.2f}% of {basis})")
    print(f"urban % of {basis:<8}: {report['urban_pct_of_study']:.2f}%")
    print(f"rural % of {basis:<8}: {report['rural_pct_of_study']:.2f}%")
    print(f"urban & rural == 0: {report['overlap_pixels'] == 0}")
    print("rural classes used: " + ", ".join(
        f"{c}={legend.get(c, '?')}" for c in RURAL_CLASSES
    ))
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"WARN: {w}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(f"Validation failed with {len(report['errors'])} error(s).")
    print("\nOK: masks written and validated.")
    return metadata


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())