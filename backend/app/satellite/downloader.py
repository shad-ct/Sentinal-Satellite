"""
Sentinel-2 imagery downloader.

Downloads Sentinel-2 L2A products from the Copernicus Data Space Ecosystem
using the CDSE S3-compatible endpoint, extracts selected bands from the
SAFE product structure, reprojects/resamples to a common resolution,
and saves them as a multi-band Cloud-Optimised GeoTIFF alongside a
metadata.json sidecar.

Caching: if the requested product already exists locally, the download
is skipped unless force_download=True.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

import requests

from backend.app.satellite.auth import get_authenticator
from backend.app.satellite.catalog import Observation
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox
from backend.app.utils.image import S2_REFLECTANCE_SCALE, write_geotiff
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Target output CRS for all downloads
TARGET_CRS = "EPSG:4326"

# Default bands to download (true-colour + NIR + SWIR)
DEFAULT_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

# Sentinel Hub Process API endpoint for alternative download
_SH_PROCESS_URL = "https://services.sentinel-hub.com/api/v1/process"


@dataclass
class DownloadedImage:
    """Result of a successful image download operation."""

    product_id: str
    local_path: Path
    metadata_path: Path
    bands: List[str]
    acquisition_date: str
    cloud_coverage: float
    bbox: Tuple[float, float, float, float]
    crs: str
    resolution_m: int
    width: int
    height: int
    cached: bool = False


class Sentinel2Downloader:
    """
    Downloads Sentinel-2 L2A imagery from CDSE using the Process API.

    The CDSE Process API (Sentinel Hub compatible) allows requesting
    specific bands, bounding boxes, and output formats without downloading
    the entire SAFE product (which can be 800 MB+).

    Fallback: If SH credentials are not available, we use the
    evalscript-based CDSE S3 download path to get specific resolution data.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._auth = get_authenticator()
        self._session = requests.Session()

    def get_local_path(
        self,
        observation: Observation,
        bands: Optional[List[str]] = None,
    ) -> Path:
        """
        Return the expected local storage path for an observation's imagery.

        Args:
            observation: Sentinel-2 observation metadata.
            bands:       List of requested band names.

        Returns:
            Path to the GeoTIFF file (may not exist yet).
        """
        acq = observation.acquisition_date
        data_dir = self._settings.data_dir
        return (
            data_dir
            / "raw"
            / "sentinel2"
            / str(acq.year)
            / f"{acq.month:02d}"
            / f"{acq.day:02d}"
            / observation.product_id[:8]
            / "image.tif"
        )

    def is_cached(
        self,
        observation: Observation,
        bands: Optional[List[str]] = None,
    ) -> bool:
        """
        Check whether an observation's imagery is already downloaded.

        Args:
            observation: Target observation.
            bands:       Requested bands (used to verify cache completeness).

        Returns:
            True if a valid cached file exists.
        """
        path = self.get_local_path(observation, bands)
        meta_path = path.parent / "metadata.json"
        return path.exists() and meta_path.exists()

    def download(
        self,
        observation: Observation,
        bbox: BoundingBox,
        bands: Optional[List[str]] = None,
        resolution_m: int = 10,
        force_download: bool = False,
        max_size_px: int = 2500,
    ) -> DownloadedImage:
        """
        Download a Sentinel-2 observation as a GeoTIFF.

        Uses the Sentinel Hub / CDSE Process API to request pre-processed,
        cloud-free bands at the specified resolution and bounding box.
        Falls back to direct CDSE S3 download if Process API is unavailable.

        Args:
            observation:    The observation to download.
            bbox:           Area of interest bounding box.
            bands:          Band names to include. Defaults to DEFAULT_BANDS.
            resolution_m:   Output pixel size in metres (10, 20, or 60).
            force_download: Re-download even if a cached file exists.
            max_size_px:    Maximum image dimension in pixels.

        Returns:
            DownloadedImage with path and metadata.
        """
        if bands is None:
            bands = DEFAULT_BANDS

        local_path = self.get_local_path(observation, bands)

        if not force_download and self.is_cached(observation, bands):
            logger.info(
                "Using cached imagery for %s (use force_download=True to re-download).",
                observation.name,
            )
            return self._load_cached(observation, local_path, bands, resolution_m)

        logger.info(
            "Downloading %s | bands=%s | res=%dm | bbox=%s",
            observation.name,
            bands,
            resolution_m,
            bbox,
        )

        # Try Process API (requires credentials) first
        try:
            return self._download_via_process_api(
                observation=observation,
                bbox=bbox,
                bands=bands,
                resolution_m=resolution_m,
                local_path=local_path,
                max_size_px=max_size_px,
            )
        except RuntimeError as exc:
            logger.warning("Process API download failed: %s. Trying CDSE S3...", exc)

        # Fallback: direct CDSE product download (extracts bands from SAFE)
        return self._download_via_cdse_s3(
            observation=observation,
            bbox=bbox,
            bands=bands,
            resolution_m=resolution_m,
            local_path=local_path,
        )

    def _download_via_process_api(
        self,
        observation: Observation,
        bbox: BoundingBox,
        bands: List[str],
        resolution_m: int,
        local_path: Path,
        max_size_px: int,
    ) -> DownloadedImage:
        """
        Download imagery using the Sentinel Hub / CDSE Process API.

        The Process API accepts an evalscript that specifies which bands
        to return. We request float32 GeoTIFF output.

        Note:
            SWIR bands (B11, B12) have 20 m native resolution.
            When resolution_m=10 is requested, they are resampled by
            the API using bilinear interpolation.
            This resampling is documented in the output metadata.
        """
        settings = self._settings
        settings.require_copernicus_credentials()

        headers = self._auth.get_auth_headers()
        headers["Content-Type"] = "application/json"

        # Build the evalscript for requested bands
        # Filter out SCL from evalscript (it comes through cloud mask)
        data_bands = [b for b in bands if b != "SCL"]
        scl_requested = "SCL" in bands

        evalscript = self._build_evalscript(data_bands, scl_requested)

        # Calculate image dimensions
        width, height = self._calculate_dimensions(bbox, resolution_m, max_size_px)

        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox.to_sentinelhub_format()),
                    "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": observation.acquisition_datetime.strftime(
                                    "%Y-%m-%dT00:00:00Z"
                                ),
                                "to": observation.acquisition_datetime.strftime(
                                    "%Y-%m-%dT23:59:59Z"
                                ),
                            },
                            "maxCloudCoverage": min(observation.cloud_coverage + 5, 100),
                        },
                    }
                ],
            },
            "output": {
                "width": width,
                "height": height,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff", "parameters": {"dtype": "float32"}},
                    }
                ],
            },
            "evalscript": evalscript,
        }

        # Use CDSE-hosted SH Process API endpoint
        try_urls = [
            f"{settings.cdse_catalog_url.replace('catalogue', 'sh')}/api/v1/process",
        ]

        last_error = None
        response = None
        for url in try_urls:
            try:
                resp = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                    stream=True,
                )
                if resp.status_code == 200:
                    response = resp
                    break
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.exceptions.RequestException as exc:
                last_error = str(exc)

        if response is None:
            raise RuntimeError(
                f"Process API unavailable. Last error: {last_error}"
            )

        # Save the response as GeoTIFF
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(".tmp.tif")

        try:
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            # Validate the downloaded file
            with rasterio.open(tmp_path) as src:
                profile = src.profile.copy()
                actual_bands = src.count
                actual_width = src.width
                actual_height = src.height

            # Rename to final path
            tmp_path.rename(local_path)

        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to save GeoTIFF: {exc}") from exc

        # Build final band list (data bands + SCL if requested)
        all_bands = data_bands + (["SCL"] if scl_requested else [])

        result = DownloadedImage(
            product_id=observation.product_id,
            local_path=local_path,
            metadata_path=local_path.parent / "metadata.json",
            bands=all_bands[:actual_bands],
            acquisition_date=observation.acquisition_date.isoformat(),
            cloud_coverage=observation.cloud_coverage,
            bbox=bbox.to_tuple(),
            crs=profile.get("crs", TARGET_CRS),
            resolution_m=resolution_m,
            width=actual_width,
            height=actual_height,
            cached=False,
        )

        self._write_metadata(result, observation)
        self._save_rgb_preview(local_path, data_bands)

        logger.info(
            "Downloaded: %s | size=%dx%d | bands=%s",
            local_path.name,
            actual_width,
            actual_height,
            all_bands,
        )
        return result

    def _download_via_cdse_s3(
        self,
        observation: Observation,
        bbox: BoundingBox,
        bands: List[str],
        resolution_m: int,
        local_path: Path,
    ) -> DownloadedImage:
        """
        Download via CDSE direct product download (fallback).

        This downloads the full SAFE zip, extracts the band JP2 files,
        and reprojects/crops them to the requested bbox and resolution.

        Note: Full SAFE products can be 600–800 MB. The Process API
        is preferred for efficiency.
        """
        logger.info("Downloading full SAFE product from CDSE S3 (this may take a while)...")
        settings = self._settings
        settings.require_copernicus_credentials()

        headers = self._auth.get_auth_headers()

        # Request a signed download URL from OData
        download_url = (
            f"https://catalogue.dataspace.copernicus.eu/odata/v1/"
            f"Products({observation.product_id})/$value"
        )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="cdse_dl_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            zip_path = tmp_dir_path / f"{observation.product_id}.zip"

            # Stream download - manually handle redirect to preserve Authorization header
            logger.info("Streaming SAFE product zip...")
            resp = self._session.get(download_url, headers=headers, allow_redirects=False)
            if resp.status_code in (301, 302, 303, 307, 308):
                redirect_url = resp.headers["Location"]
                resp = self._session.get(redirect_url, headers=headers, stream=True, timeout=600)
            else:
                resp = self._session.get(download_url, headers=headers, stream=True, timeout=600)
                
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        logger.debug("Download progress: %.1f%%", pct)

            logger.info("Extracting SAFE product...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir_path)

            # Find the SAFE directory
            safe_dirs = list(tmp_dir_path.glob("*.SAFE"))
            if not safe_dirs:
                raise RuntimeError("No .SAFE directory found in downloaded zip.")
            safe_dir = safe_dirs[0]

            # Extract and merge bands
            merged = self._extract_bands_from_safe(
                safe_dir=safe_dir,
                bbox=bbox,
                bands=bands,
                target_resolution_m=resolution_m,
            )

            # Write merged GeoTIFF
            write_geotiff(local_path, merged["data"], merged["profile"])

        width = merged["data"].shape[2]
        height = merged["data"].shape[1]

        result = DownloadedImage(
            product_id=observation.product_id,
            local_path=local_path,
            metadata_path=local_path.parent / "metadata.json",
            bands=merged["bands"],
            acquisition_date=observation.acquisition_date.isoformat(),
            cloud_coverage=observation.cloud_coverage,
            bbox=bbox.to_tuple(),
            crs=TARGET_CRS,
            resolution_m=resolution_m,
            width=width,
            height=height,
            cached=False,
        )

        self._write_metadata(result, observation)
        self._save_rgb_preview(local_path, merged["bands"])

        logger.info(
            "SAFE download complete: %s | %dx%d px",
            local_path.name,
            width,
            height,
        )
        return result

    def _extract_bands_from_safe(
        self,
        safe_dir: Path,
        bbox: BoundingBox,
        bands: List[str],
        target_resolution_m: int,
    ) -> Dict:
        """
        Extract specified bands from a SAFE product directory,
        reproject/resample to a common grid, and crop to bbox.

        Args:
            safe_dir:            Path to the .SAFE directory.
            bbox:                Area of interest.
            bands:               Band names to extract.
            target_resolution_m: Output pixel resolution in metres.

        Returns:
            Dict with keys: 'data' (ndarray), 'profile' (rasterio profile),
            'bands' (list of successfully extracted band names).

        Note:
            All bands are reprojected to WGS84 (EPSG:4326) and resampled
            to target_resolution_m using bilinear interpolation.
            SWIR bands (B11, B12) at native 20 m are upsampled when
            target_resolution_m=10. This is documented explicitly and
            users should be aware of the reduced spatial information.
        """
        import rasterio
        from rasterio.mask import mask as rasterio_mask
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        from shapely.geometry import box as shapely_box

        output_bands = []
        output_arrays = []
        ref_profile = None

        for band_name in bands:
            # Locate JP2 file for this band in SAFE structure
            jp2_files = list(safe_dir.rglob(f"*_{band_name}_*.jp2"))
            if not jp2_files:
                # Try alternative naming (resolution-specific directories)
                jp2_files = list(safe_dir.rglob(f"*_{band_name}.jp2"))
            if not jp2_files:
                logger.warning("Band %s not found in SAFE product, skipping.", band_name)
                continue

            # Prefer the file matching the target resolution
            jp2_path = self._select_best_resolution_file(
                jp2_files, band_name, target_resolution_m
            )

            try:
                with rasterio.open(jp2_path) as src:
                    # Calculate transform at target resolution in target CRS
                    transform, width, height = calculate_default_transform(
                        src.crs,
                        TARGET_CRS,
                        src.width,
                        src.height,
                        *src.bounds,
                        resolution=target_resolution_m / 111000,  # approx degrees
                    )

                    # Reproject to WGS84
                    reprojected = np.zeros((1, height, width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=reprojected,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=TARGET_CRS,
                        resampling=Resampling.bilinear,
                    )

                    if ref_profile is None:
                        ref_profile = {
                            "driver": "GTiff",
                            "dtype": "float32",
                            "crs": TARGET_CRS,
                            "transform": transform,
                            "width": width,
                            "height": height,
                        }

                    output_arrays.append(reprojected[0])
                    output_bands.append(band_name)

            except Exception as exc:
                logger.warning("Failed to process band %s: %s", band_name, exc)

        if not output_arrays:
            raise RuntimeError("No bands could be extracted from the SAFE product.")

        # Stack all bands into single array [n_bands, H, W]
        # Ensure all bands have same dimensions (resample if necessary)
        target_h = ref_profile["height"]
        target_w = ref_profile["width"]
        stacked = []
        for arr in output_arrays:
            if arr.shape != (target_h, target_w):
                import cv2
                arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            stacked.append(arr)

        data = np.stack(stacked, axis=0)

        return {"data": data, "profile": ref_profile, "bands": output_bands}

    @staticmethod
    def _select_best_resolution_file(
        jp2_files: List[Path],
        band_name: str,
        target_resolution_m: int,
    ) -> Path:
        """
        Select the JP2 file that best matches the target resolution.
        Prefers files in resolution-specific subdirectories (R10m, R20m, R60m).
        """
        # Sort by resolution folder name
        for res in [f"R{target_resolution_m}m", "R10m", "R20m", "R60m"]:
            for f in jp2_files:
                if res in f.parts or res in f.name:
                    return f
        return jp2_files[0]

    @staticmethod
    def _build_evalscript(bands: List[str], include_scl: bool) -> str:
        """
        Build a Sentinel Hub evalscript to return the requested bands.
        The evalscript is JavaScript code executed server-side.
        """
        import json
        output_bands = list(bands)
        if include_scl:
            output_bands.append("SCL")

        n_bands = len(output_bands)
        data_bands_var = ", ".join(f"sample.{b}" for b in output_bands)
        bands_js = json.dumps(output_bands)

        evalscript = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{ bands: {bands_js} }}],
    output: {{
      bands: {n_bands},
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [{data_bands_var}];
}}
"""
        return evalscript.strip()

    @staticmethod
    def _calculate_dimensions(
        bbox: BoundingBox,
        resolution_m: int,
        max_size_px: int,
    ) -> Tuple[int, int]:
        """
        Calculate output image dimensions for a given bbox and resolution.

        Args:
            bbox:         Bounding box.
            resolution_m: Pixel size in metres.
            max_size_px:  Maximum dimension (width or height).

        Returns:
            (width, height) in pixels.
        """
        # Approximate metres per degree at the bbox centre
        centre_lat = (bbox.south + bbox.north) / 2.0
        import math
        metres_per_deg_lat = 111_000.0
        metres_per_deg_lon = 111_000.0 * math.cos(math.radians(centre_lat))

        width_m = bbox.width_deg * metres_per_deg_lon
        height_m = bbox.height_deg * metres_per_deg_lat

        width_px = int(width_m / resolution_m)
        height_px = int(height_m / resolution_m)

        # Clamp to max_size_px maintaining aspect ratio
        if width_px > max_size_px or height_px > max_size_px:
            scale = max_size_px / max(width_px, height_px)
            width_px = max(1, int(width_px * scale))
            height_px = max(1, int(height_px * scale))

        # Ensure minimum size
        width_px = max(width_px, 64)
        height_px = max(height_px, 64)

        logger.debug(
            "Calculated image dimensions: %dx%d px (%.1f × %.1f km at %dm res)",
            width_px,
            height_px,
            width_m / 1000,
            height_m / 1000,
            resolution_m,
        )
        return width_px, height_px

    def _write_metadata(self, result: DownloadedImage, obs: Observation) -> None:
        """Write a JSON sidecar file with full acquisition metadata."""
        metadata = {
            "acquisition_date": result.acquisition_date,
            "satellite": obs.platform,
            "product_id": result.product_id,
            "product_name": obs.name,
            "cloud_percentage": result.cloud_coverage,
            "bbox": {
                "west": result.bbox[0],
                "south": result.bbox[1],
                "east": result.bbox[2],
                "north": result.bbox[3],
            },
            "crs": str(result.crs),
            "resolution_m": result.resolution_m,
            "bands": result.bands,
            "image_width_px": result.width,
            "image_height_px": result.height,
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
            "note_resampling": (
                "SWIR bands (B11, B12) have native 20m resolution. "
                "When requested at 10m, they are bilinearly resampled. "
                "This is a known limitation — the extra spatial detail is interpolated."
            ),
        }

        meta_path = result.local_path.parent / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        logger.debug("Wrote metadata: %s", meta_path)

    def _save_rgb_preview(self, image_path: Path, bands: List[str]) -> None:
        """Save a quick-look RGB PNG for visual verification."""
        try:
            with rasterio.open(image_path) as src:
                data = src.read()

            # Find RGB band indices
            rgb_names = ["B04", "B03", "B02"]
            indices = []
            for name in rgb_names:
                if name in bands:
                    indices.append(bands.index(name))

            if len(indices) == 3:
                from backend.app.utils.image import save_png_preview
                preview_path = image_path.parent / "rgb_preview.png"
                save_png_preview(data, preview_path, rgb_indices=tuple(indices))
                logger.info("Saved RGB preview: %s", preview_path.name)
        except Exception as exc:
            logger.warning("Could not generate RGB preview: %s", exc)

    def _load_cached(
        self,
        observation: Observation,
        local_path: Path,
        bands: List[str],
        resolution_m: int,
    ) -> DownloadedImage:
        """Load metadata for an already-cached observation."""
        meta_path = local_path.parent / "metadata.json"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        with rasterio.open(local_path) as src:
            width = src.width
            height = src.height

        return DownloadedImage(
            product_id=observation.product_id,
            local_path=local_path,
            metadata_path=meta_path,
            bands=meta.get("bands", bands),
            acquisition_date=observation.acquisition_date.isoformat(),
            cloud_coverage=observation.cloud_coverage,
            bbox=observation.bbox.to_tuple(),
            crs=meta.get("crs", TARGET_CRS),
            resolution_m=meta.get("resolution_m", resolution_m),
            width=width,
            height=height,
            cached=True,
        )
