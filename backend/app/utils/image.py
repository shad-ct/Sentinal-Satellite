"""
Image I/O utilities for Sentinel-2 imagery.

Handles reading/writing GeoTIFF files with rasterio, generating
thumbnails, normalising band data, and converting between formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# Sentinel-2 reflectance scale factor (L2A products store DN as uint16)
# Divide by this to obtain approximate surface reflectance (0–1 range)
S2_REFLECTANCE_SCALE = 10000.0

# Sentinel-2 band → native resolution mapping (metres)
S2_BAND_RESOLUTION: Dict[str, int] = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
    "SCL": 20,  # Scene Classification Layer
    "TCI": 10,  # True Colour Image
}

# RGB band combination for true-colour composite
RGB_BANDS = ["B04", "B03", "B02"]  # Red, Green, Blue


def read_geotiff(
    path: Union[str, Path],
    bands: Optional[List[int]] = None,
) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    """
    Read a GeoTIFF file and return its data and profile.

    Args:
        path:  Path to the GeoTIFF file.
        bands: 1-indexed list of band indices to read.
               If None, all bands are read.

    Returns:
        Tuple of (data array with shape [bands, height, width], rasterio profile).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        profile = src.profile.copy()
        if bands is None:
            data = src.read()
        else:
            data = src.read(bands)

    logger.debug(
        "Read GeoTIFF: %s | shape=%s | dtype=%s | CRS=%s",
        path.name,
        data.shape,
        data.dtype,
        profile.get("crs"),
    )
    return data, profile


def write_geotiff(
    path: Union[str, Path],
    data: np.ndarray,
    profile: rasterio.profiles.Profile,
) -> None:
    """
    Write a numpy array to a GeoTIFF file using an existing rasterio profile.

    Args:
        path:    Output file path.
        data:    Array of shape [bands, height, width].
        profile: Rasterio profile dict (CRS, transform, dtype, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Add band dimension

    updated_profile = profile.copy()
    updated_profile.update(
        count=data.shape[0],
        dtype=data.dtype,
        driver="GTiff",
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(path, "w", **updated_profile) as dst:
        dst.write(data)

    logger.debug("Wrote GeoTIFF: %s | shape=%s", path.name, data.shape)


def normalize_band(
    band: np.ndarray,
    scale: float = S2_REFLECTANCE_SCALE,
    clip_max: float = 1.0,
) -> np.ndarray:
    """
    Normalize a Sentinel-2 band from uint16 DN to float32 reflectance.

    Args:
        band:     Input array (uint16 DN values).
        scale:    Division factor (default: 10000 for S2 L2A).
        clip_max: Upper clip value after scaling.

    Returns:
        Float32 array in range [0, clip_max].
    """
    result = band.astype(np.float32) / scale
    return np.clip(result, 0.0, clip_max)


def normalize_for_display(
    data: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    Stretch array values for display using percentile clipping.

    Args:
        data:   Input array (any dtype).
        p_low:  Lower percentile for clipping.
        p_high: Upper percentile for clipping.

    Returns:
        Float32 array with values in [0, 1].
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))

    if hi == lo:
        return np.zeros_like(data, dtype=np.float32)

    stretched = (data.astype(np.float32) - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0)


def bands_to_rgb(
    bands: np.ndarray,
    band_indices: Tuple[int, int, int] = (0, 1, 2),
    stretch: bool = True,
) -> np.ndarray:
    """
    Convert multi-band array to uint8 RGB image array.

    Args:
        bands:       Array of shape [n_bands, H, W].
        band_indices: Indices into bands for (R, G, B).
        stretch:     Apply percentile stretch for better display.

    Returns:
        uint8 array of shape [H, W, 3] suitable for PIL/OpenCV.
    """
    r = bands[band_indices[0]]
    g = bands[band_indices[1]]
    b = bands[band_indices[2]]

    if stretch:
        r = normalize_for_display(r)
        g = normalize_for_display(g)
        b = normalize_for_display(b)

    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3) float [0, 1]
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgb_uint8


def save_png_preview(
    data: np.ndarray,
    path: Union[str, Path],
    rgb_indices: Tuple[int, int, int] = (0, 1, 2),
) -> None:
    """
    Save a multi-band float/uint16 array as a PNG preview.

    Args:
        data:       Array of shape [n_bands, H, W].
        path:       Output PNG path.
        rgb_indices: Band indices for R, G, B.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = bands_to_rgb(data, rgb_indices, stretch=True)
    Image.fromarray(rgb).save(path)
    logger.debug("Saved PNG preview: %s", path)


def save_colormap_png(
    data: np.ndarray,
    path: Union[str, Path],
    colormap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Save a single-band float array as a colourmap PNG.

    Args:
        data:     2D float array.
        path:     Output PNG path.
        colormap: Matplotlib colourmap name.
        vmin:     Minimum value for colour mapping.
        vmax:     Maximum value for colour mapping.
    """
    import matplotlib.pyplot as plt  # Deferred import to avoid startup cost

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    logger.debug("Saved colormap PNG: %s", path)


def array_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Normalize and convert a float array to uint8 [0, 255].
    Handles NaN values by setting them to 0.
    """
    arr = np.nan_to_num(array, nan=0.0)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * 255).astype(np.uint8)
