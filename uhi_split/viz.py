"""Diagnostic visualizations for the spatial split and tensor samples."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .split import TRAIN, VAL, TEST, SPLIT_NAMES


def _colors():
    return {
        TRAIN: (0.20, 0.55, 0.90),
        VAL: (0.95, 0.75, 0.15),
        TEST: (0.85, 0.25, 0.25),
        0: (0.92, 0.92, 0.92),
    }


def plot_spatial_split(
    split_mask: np.ndarray,
    region_mask: np.ndarray,
    stats: dict,
    out_path: Path,
    boundary=None,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    H, W = split_mask.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    for sid, col in _colors().items():
        img[split_mask == sid] = col

    fig, ax = plt.subplots(figsize=(10, 10))
    if bounds is not None:
        left, right, bottom, top = bounds
        ax.imshow(img, extent=(left, right, bottom, top), origin="upper")
    else:
        ax.imshow(img)
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="black", linewidth=0.4)
    pct = stats["pixel_pct"]
    handles = [
        Patch(color=_colors()[TRAIN], label=f"TRAIN (Central+South)  {pct['train']:.1f}%"),
        Patch(color=_colors()[VAL], label=f"VAL (East)  {pct['val']:.1f}%"),
        Patch(color=_colors()[TEST], label=f"TEST (North+buffer)  {pct['test']:.1f}%"),
        Patch(color=_colors()[0], label="outside modeling area"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.set_title(
        "Spatial split (artificial directional-quantile partition; NOT official admin units)\n"
        f"method={stats['split_method']} rotation={stats['rotation_deg']}deg "
        f"test={stats['test_fraction']} val={stats['val_fraction']}"
    )
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_sample_grid(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    channel_names: list[str],
    lst_idx: int,
    out_path: Path,
    target_month: str,
    input_months: list[str],
) -> None:
    """X [L,C,s,s] normalized, y [1,s,s] normalized, mask [1,s,s]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    hist_lst = X[0, lst_idx]
    im = axes[0, 0].imshow(hist_lst, cmap="inferno")
    axes[0, 0].set_title(f"Historical LST input (normalized)\n{input_months[0]}")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(y[0], cmap="inferno")
    axes[0, 1].set_title(f"Target LST (normalized)\n{target_month}")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(mask[0], cmap="gray")
    axes[0, 2].set_title("Target validity mask\n1=valid, 0=filled")

    # normalized feature example: NDVI (ch 1) from most recent input month
    ndvi_idx = channel_names.index("NDVI")
    im = axes[1, 0].imshow(X[-1, ndvi_idx], cmap="viridis")
    axes[1, 0].set_title(f"NDVI input (normalized)\n{input_months[-1]}")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # per-channel mean over the patch, all 18 channels
    means = X.reshape(X.shape[0], X.shape[1], -1).mean(axis=(0, 2))
    axes[1, 1].barh(range(len(channel_names)), means, color="steelblue")
    axes[1, 1].set_yticks(range(len(channel_names)))
    axes[1, 1].set_yticklabels(channel_names, fontsize=7)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("Per-channel patch mean (normalized)")

    # distribution of all normalized channels
    axes[1, 2].hist(X.ravel(), bins=100, color="slateblue")
    axes[1, 2].set_title("Normalized X histogram (all channels)")

    for ax in axes.ravel()[:5]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Diagnostic sample: 12 historical months -> following month")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_split_profiles(
    split_mask: np.ndarray,
    out_path: Path,
    row_index: int | None = None,
    col_index: int | None = None,
) -> None:
    """Row/column profile of split ids to show contiguity of the partition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = split_mask.shape
    if row_index is None:
        row_index = H // 2
    if col_index is None:
        col_index = W // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(split_mask[row_index], linewidth=0.8, color="black")
    axes[0].set_title(f"Split id along row {row_index}")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["out", "train", "val", "test"])
    axes[1].plot(split_mask[:, col_index], linewidth=0.8, color="black")
    axes[1].set_title(f"Split id along column {col_index}")
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["out", "train", "val", "test"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)