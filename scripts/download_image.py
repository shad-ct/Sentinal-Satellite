#!/usr/bin/env python3
"""
CLI: Download a Sentinel-2 image for a given location and date.

Usage:
  python scripts/download_image.py --lat 11.8745 --lon 75.3704 --date 2024-06-15
  python scripts/download_image.py --lat 11.8745 --lon 75.3704 --date 2024-06-15 --size-km 20 --cloud 30
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from datetime import date

from backend.app.satellite.catalog import CDSECatalog
from backend.app.satellite.downloader import Sentinel2Downloader
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox
from backend.app.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@click.command()
@click.option("--lat", type=float, required=True, help="Centre latitude")
@click.option("--lon", type=float, required=True, help="Centre longitude")
@click.option("--date", "target_date", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--size-km", type=float, default=10.0, help="AOI side length in km")
@click.option("--cloud", type=float, default=20.0, help="Max cloud coverage (%)")
@click.option("--window", type=int, default=15, help="Search window (±days)")
@click.option("--resolution", type=int, default=10, help="Output resolution (m)")
@click.option("--bands", default="B02,B03,B04,B08,B11,B12,SCL", help="Comma-separated bands")
@click.option("--strategy", type=click.Choice(["nearest", "clearest"]), default="nearest")
@click.option("--force", is_flag=True, help="Force re-download if cached")
def download_image(lat, lon, target_date, size_km, cloud, window, resolution, bands, strategy, force):
    """Download a Sentinel-2 L2A image for a given location and date."""

    settings = get_settings()
    try:
        settings.require_copernicus_credentials()
    except RuntimeError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    bbox = BoundingBox.from_center(lat, lon, size_km)
    band_list = [b.strip() for b in bands.split(",")]
    target = target_date.date() if hasattr(target_date, "date") else target_date

    click.echo(f"\n🛰  Sentinel-2 Image Download")
    click.echo(f"   Location: lat={lat}, lon={lon}")
    click.echo(f"   Target date: {target} (±{window} days)")
    click.echo(f"   Max cloud: {cloud}%  |  Strategy: {strategy}")
    click.echo(f"   AOI: {bbox}")
    click.echo(f"   Bands: {band_list}\n")

    catalog = CDSECatalog()
    obs = catalog.get_best_observation(
        bbox=bbox,
        target_date=target,
        window_days=window,
        max_cloud_coverage=cloud,
        strategy=strategy,
    )

    if obs is None:
        click.echo("❌  No observations found. Try increasing --window or --cloud.", err=True)
        sys.exit(1)

    click.echo(f"✅  Selected: {obs.name}")
    click.echo(f"   Date: {obs.acquisition_date}  |  Cloud: {obs.cloud_coverage:.1f}%")

    downloader = Sentinel2Downloader()
    result = downloader.download(
        observation=obs,
        bbox=bbox,
        bands=band_list,
        resolution_m=resolution,
        force_download=force,
    )

    click.echo(f"\n✅  Download complete!")
    click.echo(f"   GeoTIFF: {result.local_path}")
    click.echo(f"   Metadata: {result.metadata_path}")
    click.echo(f"   Size: {result.width}×{result.height} px  |  Bands: {result.bands}")
    if result.cached:
        click.echo("   ℹ️  Used cached file (use --force to re-download)")


if __name__ == "__main__":
    download_image()
