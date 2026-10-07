"""
Change map visualization generator.

Combines the change score, binary mask, and index results into a set
of rich visualizations saved as PNG files.

Visualization outputs:
  1. before_rgb.png     — true-colour before image
  2. after_rgb.png      — true-colour after image
  3. change_score.png   — continuous change probability (jet colormap)
  4. change_mask.png    — binary change overlay on RGB
  5. ndvi_before.png    — NDVI before (RdYlGn colormap)
  6. ndvi_after.png     — NDVI after
  7. ndvi_diff.png      — NDVI change (diverging colormap)
  8. composite.png      — 2×4 summary grid

Color settings are defined in one place (VISUALIZATION_CONFIG) so that
they can be tuned without modifying function logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from backend.app.utils.image import bands_to_rgb, normalize_for_display
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


# ── Centralized visualization configuration ───────────────────────────────────
VISUALIZATION_CONFIG = {
    "change_score_colormap": "hot",
    "ndvi_colormap": "RdYlGn",
    "ndwi_colormap": "Blues",
    "ndbi_colormap": "RdBu_r",
    "change_overlay_color": (255, 60, 0),     # Orange-red for changed regions
    "change_overlay_alpha": 0.6,
    "figure_dpi": 120,
    "thumbnail_max_px": 512,
}


def generate_visualizations(
    before: np.ndarray,
    after: np.ndarray,
    change_score: np.ndarray,
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    output_dir: Path,
    band_names: Optional[list[str]] = None,
    index_results: Optional[Dict] = None,
) -> Dict[str, Path]:
    """
    Generate all change detection visualizations and save to output_dir.

    Args:
        before:       Before image [n_bands, H, W] (float32 reflectance).
        after:        After image [n_bands, H, W] (float32 reflectance).
        change_score: Continuous change score [H, W] in [0, 1].
        binary_mask:  Binary change mask [H, W] (0 or 1).
        valid_mask:   Boolean valid pixel mask [H, W].
        output_dir:   Directory to save PNG files.
        band_names:   Band names matching first axis (used to find RGB, NDVI bands).
        index_results: Dict of IndexResult objects (NDVI, NDWI, etc.).

    Returns:
        Dict mapping output name → saved Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    if band_names is None:
        band_names = [f"B{i+1}" for i in range(before.shape[0])]

    # Find RGB band indices
    rgb_idx = _find_rgb_indices(band_names)

    # ── 1. Before RGB ──────────────────────────────────────────────────────
    before_rgb = bands_to_rgb(before, rgb_idx, stretch=True)
    saved["before_rgb"] = _save_single(before_rgb, output_dir / "before_rgb.png", "Before")

    # ── 2. After RGB ───────────────────────────────────────────────────────
    after_rgb = bands_to_rgb(after, rgb_idx, stretch=True)
    saved["after_rgb"] = _save_single(after_rgb, output_dir / "after_rgb.png", "After")

    # ── 3. Change Score Map ────────────────────────────────────────────────
    saved["change_score"] = _save_colormap(
        data=np.where(valid_mask, change_score, np.nan),
        path=output_dir / "change_score.png",
        title="Change Score",
        cmap=VISUALIZATION_CONFIG["change_score_colormap"],
        vmin=0,
        vmax=1,
    )

    # ── 4. Binary Change Overlay ───────────────────────────────────────────
    saved["change_mask"] = _save_change_overlay(
        rgb=after_rgb,
        binary_mask=binary_mask,
        valid_mask=valid_mask,
        path=output_dir / "change_mask.png",
    )

    # ── 5–7. Index Visualizations ──────────────────────────────────────────
    if index_results:
        if "NDVI" in index_results:
            ndvi_before = index_results["NDVI"]
            saved["ndvi_before"] = _save_colormap(
                data=np.where(ndvi_before.valid_mask, ndvi_before.values, np.nan),
                path=output_dir / "ndvi_before.png",
                title="NDVI (Before)",
                cmap=VISUALIZATION_CONFIG["ndvi_colormap"],
                vmin=-1,
                vmax=1,
            )

        if "NDVI_after" in index_results:
            ndvi_after = index_results["NDVI_after"]
            saved["ndvi_after"] = _save_colormap(
                data=np.where(ndvi_after.valid_mask, ndvi_after.values, np.nan),
                path=output_dir / "ndvi_after.png",
                title="NDVI (After)",
                cmap=VISUALIZATION_CONFIG["ndvi_colormap"],
                vmin=-1,
                vmax=1,
            )

        if "NDVI_change" in index_results:
            ndvi_diff = index_results["NDVI_change"]
            saved["ndvi_diff"] = _save_colormap(
                data=np.where(ndvi_diff.valid_mask, ndvi_diff.values, np.nan),
                path=output_dir / "ndvi_diff.png",
                title="NDVI Change (After − Before)",
                cmap="RdBu",
                vmin=-0.5,
                vmax=0.5,
            )

    # ── 8. Composite Summary Grid ──────────────────────────────────────────
    saved["composite"] = _save_composite(
        saved_paths=saved,
        output_path=output_dir / "composite.png",
    )

    logger.info(
        "Generated %d visualization files in %s",
        len(saved),
        output_dir,
    )
    return saved


def _find_rgb_indices(band_names: list[str]) -> tuple[int, int, int]:
    """Find (R, G, B) indices from band_names. Defaults to (0, 1, 2)."""
    rgb_map = {"B04": 0, "B03": 1, "B02": 2}
    indices = []
    for band in ["B04", "B03", "B02"]:
        if band in band_names:
            indices.append(band_names.index(band))
    if len(indices) == 3:
        return tuple(indices)
    # Fallback: use first three bands
    n = len(band_names)
    return (0, min(1, n - 1), min(2, n - 1))


def _save_single(rgb: np.ndarray, path: Path, title: str = "") -> Path:
    """Save a uint8 RGB array as PNG."""
    from PIL import Image
    img = Image.fromarray(rgb)
    img.save(path)
    logger.debug("Saved: %s", path.name)
    return path


def _save_colormap(
    data: np.ndarray,
    path: Path,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """Save a 2D float array as a colourmap PNG with title and colorbar."""
    dpi = VISUALIZATION_CONFIG["figure_dpi"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug("Saved colormap: %s", path.name)
    return path


def _save_change_overlay(
    rgb: np.ndarray,
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    path: Path,
) -> Path:
    """Overlay changed regions in red-orange on the after RGB image."""
    overlay = rgb.astype(np.float32).copy()
    changed = (binary_mask == 1) & valid_mask
    color = VISUALIZATION_CONFIG["change_overlay_color"]
    alpha = VISUALIZATION_CONFIG["change_overlay_alpha"]

    for c, channel_val in enumerate(color):
        overlay[..., c] = np.where(
            changed,
            (1 - alpha) * overlay[..., c] + alpha * channel_val,
            overlay[..., c],
        )

    from PIL import Image
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(path)
    logger.debug("Saved change overlay: %s", path.name)
    return path


def _save_composite(
    saved_paths: Dict[str, Path],
    output_path: Path,
) -> Path:
    """Create a composite summary grid of all generated images."""
    keys_to_show = [
        ("before_rgb", "Before"),
        ("after_rgb", "After"),
        ("change_score", "Change Score"),
        ("change_mask", "Change Overlay"),
        ("ndvi_before", "NDVI Before"),
        ("ndvi_after", "NDVI After"),
        ("ndvi_diff", "NDVI Δ"),
    ]

    available = [(k, t) for k, t in keys_to_show if k in saved_paths]
    n = len(available)
    if n == 0:
        return output_path

    cols = 4
    rows = (n + cols - 1) // cols
    dpi = VISUALIZATION_CONFIG["figure_dpi"]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")

    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx, (key, title) in enumerate(available):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        img_path = saved_paths[key]

        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            ax.imshow(np.array(img))
            ax.set_title(title, fontsize=9, color="white", pad=4)
            ax.axis("off")
        except Exception as exc:
            ax.text(0.5, 0.5, f"Error: {exc}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=7)
            ax.axis("off")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].set_visible(False)

    plt.suptitle(
        "Sentinel-2 Change Detection Summary",
        fontsize=13,
        color="white",
        y=1.01,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved composite summary: %s", output_path.name)
    return output_path
