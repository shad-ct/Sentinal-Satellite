#!/usr/bin/env python3
"""
CLI: Compare two Sentinel-2 images and detect changes.

Usage:
  python scripts/compare_images.py --before data/raw/.../image.tif --after data/raw/.../image.tif
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import rasterio

from backend.app.analysis.change_map import generate_visualizations
from backend.app.analysis.difference import detect_changes
from backend.app.analysis.indices import compute_all_indices, compute_index_change
from backend.app.analysis.statistics import compute_statistics
from backend.app.preprocessing.alignment import align_images_from_files
from backend.app.preprocessing.cloud_mask import create_cloud_mask, extract_scl_from_multiband, MaskClass
from backend.app.preprocessing.normalization import normalize_band_stack
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox
from backend.app.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--before", "before_path", required=True, type=click.Path(exists=True))
@click.option("--after", "after_path", required=True, type=click.Path(exists=True))
@click.option("--threshold", type=float, default=0.15, help="Change detection threshold [0-1]")
@click.option("--sigma", type=float, default=1.5, help="Gaussian smoothing sigma")
@click.option("--min-region", type=int, default=50, help="Minimum changed region size (pixels)")
@click.option("--output-dir", default=None, help="Output directory for visualizations")
@click.option("--bands", default="B02,B03,B04,B08,B11,B12,SCL", help="Comma-separated band names")
def compare_images(before_path, after_path, threshold, sigma, min_region, output_dir, bands):
    """Run change detection between two downloaded GeoTIFF images."""

    before_path = Path(before_path)
    after_path = Path(after_path)
    band_list = [b.strip() for b in bands.split(",")]

    if output_dir is None:
        output_dir = before_path.parent / "comparison_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n🔬 Sentinel-2 Change Detection")
    click.echo(f"   Before: {before_path.name}")
    click.echo(f"   After:  {after_path.name}")
    click.echo(f"   Threshold: {threshold}  |  Sigma: {sigma}")
    click.echo()

    # Read & normalize
    click.echo("📖 Reading and normalizing images...")
    with rasterio.open(before_path) as src:
        before_raw = src.read().astype(np.float32)
        before_profile = src.profile.copy()

    with rasterio.open(after_path) as src:
        after_raw = src.read().astype(np.float32)

    before_norm = normalize_band_stack(before_raw, band_list)
    after_norm = normalize_band_stack(after_raw, band_list)

    # Cloud masks
    click.echo("☁  Applying cloud masks...")
    before_scl = extract_scl_from_multiband(before_norm, band_list)
    after_scl = extract_scl_from_multiband(after_norm, band_list)

    if before_scl is not None and after_scl is not None:
        before_cm = create_cloud_mask(before_scl)
        after_cm = create_cloud_mask(after_scl)
        valid_mask = (before_cm == MaskClass.VALID) & (after_cm == MaskClass.VALID)
        click.echo(f"   Valid pixels: {100*valid_mask.mean():.1f}%")
    else:
        valid_mask = np.ones(before_norm.shape[1:], dtype=bool)
        click.echo("   No SCL band found — skipping cloud masking")

    # Align
    click.echo("📐 Aligning images (geospatial warp)...")
    before_aligned, after_aligned, shared_profile = align_images_from_files(before_path, after_path)
    before_final = normalize_band_stack(before_aligned, band_list)
    after_final = normalize_band_stack(after_aligned, band_list)

    # Filter data bands
    data_bands = [b for b in band_list if b != "SCL"]
    data_idx = [i for i, b in enumerate(band_list) if b != "SCL"]
    before_data = before_final[data_idx]
    after_data = after_final[data_idx]

    # Spectral indices
    click.echo("📊 Computing spectral indices (NDVI, NDWI, NDBI)...")
    before_indices = compute_all_indices(before_data, data_bands)
    after_indices = compute_all_indices(after_data, data_bands)
    index_changes = {}
    all_index_results = dict(before_indices)
    for name, b_ir in before_indices.items():
        if name in after_indices:
            a_ir = after_indices[name]
            all_index_results[f"{name}_after"] = a_ir
            ch = compute_index_change(b_ir, a_ir)
            all_index_results[f"{name}_change"] = ch
            index_changes[f"{name}_change"] = ch.values
    for name, result in before_indices.items():
        mean = float(np.nanmean(result.values[result.valid_mask])) if result.valid_mask.any() else float('nan')
        click.echo(f"   {name} before: mean = {mean:.3f}")

    # Change detection
    click.echo(f"🔍 Running change detection (threshold={threshold})...")
    diff = detect_changes(
        before=before_data, after=after_data,
        valid_mask=valid_mask if valid_mask.shape == before_data.shape[1:] else None,
        threshold=threshold, gaussian_sigma=sigma, min_region_pixels=min_region,
    )

    # Try to get bbox from metadata
    meta_path = before_path.parent / "metadata.json"
    bbox = BoundingBox(-1, -1, 1, 1)  # placeholder
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        b = meta.get("bbox", {})
        if b:
            bbox = BoundingBox(b["west"], b["south"], b["east"], b["north"])

    # Statistics
    stats = compute_statistics(
        binary_mask=diff.binary_mask,
        valid_mask=diff.valid_mask,
        bbox=bbox,
        resolution_m=10,
        index_changes=index_changes,
    )

    # Visualizations
    click.echo("🎨 Generating visualizations...")
    vis = generate_visualizations(
        before=before_data, after=after_data,
        change_score=diff.change_score,
        binary_mask=diff.binary_mask,
        valid_mask=diff.valid_mask,
        output_dir=output_dir,
        band_names=data_bands,
        index_results=all_index_results,
    )

    # Summary
    click.echo()
    click.echo("=" * 50)
    click.echo("📊 CHANGE DETECTION RESULTS")
    click.echo("=" * 50)
    click.echo(f"   Total area:   {stats.total_area_km2:.3f} km²")
    click.echo(f"   Valid area:   {stats.valid_area_km2:.3f} km²")
    click.echo(f"   Changed area: {stats.changed_area_km2:.3f} km²  ({stats.change_percentage:.1f}%)")
    click.echo()
    if stats.index_hints:
        click.echo("   Spectral Index Hints (ESTIMATES ONLY):")
        for k, v in stats.index_hints.items():
            click.echo(f"     {k}: {v:.3f} km²")
    click.echo()
    click.echo(f"   ⚠  Scientific disclaimer: {stats.notes[0]}")
    click.echo()
    click.echo(f"📁 Visualizations saved to: {output_dir}")
    for name, path in vis.items():
        click.echo(f"   {name}: {path.name}")


if __name__ == "__main__":
    compare_images()
