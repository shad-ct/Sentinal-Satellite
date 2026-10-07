"""
Geospatial utility functions.

Provides bounding box validation, coordinate conversion helpers,
and CRS utilities used throughout the application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

from pyproj import CRS, Transformer
from shapely.geometry import box, mapping

from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# WGS84 geographic CRS (EPSG:4326)
WGS84 = CRS.from_epsg(4326)

# Common UTM-like projected CRS used for area calculations
# (We determine the appropriate zone dynamically)
PSEUDO_MERCATOR = CRS.from_epsg(3857)


@dataclass
class BoundingBox:
    """
    Geographic bounding box in WGS84 (EPSG:4326).

    Attributes:
        west:  Minimum longitude (°)
        south: Minimum latitude (°)
        east:  Maximum longitude (°)
        north: Maximum latitude (°)
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        validate_bounding_box(self.west, self.south, self.east, self.north)

    @classmethod
    def from_center(
        cls,
        lat: float,
        lon: float,
        size_km: float = 10.0,
    ) -> "BoundingBox":
        """
        Create a bounding box centred on (lat, lon) with a given side length.

        Args:
            lat:     Centre latitude in degrees.
            lon:     Centre longitude in degrees.
            size_km: Side length in kilometres.

        Returns:
            BoundingBox centred on the given point.
        """
        validate_coordinates(lat, lon)
        half_deg_lat = (size_km / 2.0) / 111.0  # ~111 km per degree lat
        half_deg_lon = half_deg_lat / math.cos(math.radians(lat))

        return cls(
            west=lon - half_deg_lon,
            south=lat - half_deg_lat,
            east=lon + half_deg_lon,
            north=lat + half_deg_lat,
        )

    @property
    def width_deg(self) -> float:
        """Width of the bounding box in degrees."""
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        """Height of the bounding box in degrees."""
        return self.north - self.south

    def area_km2(self) -> float:
        """
        Approximate area in km² using Web Mercator reprojection.
        Suitable for small areas; not precise for large regions.
        """
        transformer = Transformer.from_crs(WGS84, PSEUDO_MERCATOR, always_xy=True)
        x_min, y_min = transformer.transform(self.west, self.south)
        x_max, y_max = transformer.transform(self.east, self.north)
        area_m2 = (x_max - x_min) * (y_max - y_min)
        return area_m2 / 1_000_000.0

    def to_wkt(self) -> str:
        """Return WKT polygon string for use in API queries."""
        return box(self.west, self.south, self.east, self.north).wkt

    def to_geojson(self) -> dict:
        """Return GeoJSON geometry dict."""
        return mapping(box(self.west, self.south, self.east, self.north))

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return (west, south, east, north) tuple."""
        return (self.west, self.south, self.east, self.north)

    def to_sentinelhub_format(self) -> Tuple[float, float, float, float]:
        """
        Return (west, south, east, north) in the order expected by
        Sentinel Hub / rasterio: (min_lon, min_lat, max_lon, max_lat).
        """
        return (self.west, self.south, self.east, self.north)

    def __str__(self) -> str:
        return (
            f"BBox(W={self.west:.4f}, S={self.south:.4f}, "
            f"E={self.east:.4f}, N={self.north:.4f})"
        )


def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validate latitude and longitude values.

    Args:
        lat: Latitude in degrees (-90 to 90).
        lon: Longitude in degrees (-180 to 180).

    Raises:
        ValueError: If coordinates are out of range.
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(
            f"Latitude must be between -90 and 90 degrees, got {lat}"
        )
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(
            f"Longitude must be between -180 and 180 degrees, got {lon}"
        )


def validate_bounding_box(
    west: float,
    south: float,
    east: float,
    north: float,
) -> None:
    """
    Validate a bounding box.

    Args:
        west:  Minimum longitude.
        south: Minimum latitude.
        east:  Maximum longitude.
        north: Maximum latitude.

    Raises:
        ValueError: If any value is out of range or the box is degenerate.
    """
    validate_coordinates(south, west)
    validate_coordinates(north, east)

    if west >= east:
        raise ValueError(
            f"west ({west}) must be less than east ({east}). "
            "Cross-antimeridian bounding boxes are not supported."
        )
    if south >= north:
        raise ValueError(
            f"south ({south}) must be less than north ({north})."
        )

    width = east - west
    height = north - south
    if width < 0.001 or height < 0.001:
        logger.warning(
            "Very small bounding box: %.4f° × %.4f°. "
            "This may result in very few pixels at 10 m resolution.",
            width,
            height,
        )


def degrees_to_utm_epsg(lat: float, lon: float) -> int:
    """
    Determine the EPSG code of the UTM zone that contains (lat, lon).

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        EPSG code as integer (e.g. 32644 for UTM zone 44N).
    """
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone  # Northern hemisphere
    return 32700 + zone      # Southern hemisphere


def bbox_area_km2(west: float, south: float, east: float, north: float) -> float:
    """
    Convenience function to compute bounding box area without creating a BoundingBox.
    """
    return BoundingBox(west=west, south=south, east=east, north=north).area_km2()
