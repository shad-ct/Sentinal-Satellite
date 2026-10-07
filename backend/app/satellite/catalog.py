"""
Copernicus Data Space Ecosystem (CDSE) catalog search.

Searches for Sentinel-2 L2A observations using the CDSE OData / STAC API.
Returns typed Observation objects with full metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional
from urllib.parse import urlencode

import requests

from backend.app.satellite.auth import get_authenticator
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)

# CDSE OData endpoint for product catalog search
_ODATA_PRODUCTS_URL = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)

# Sentinel-2 L2A collection identifier
_S2_L2A_COLLECTION = "SENTINEL-2"
_S2_L2A_PRODUCT_TYPE = "S2MSI2A"  # Sentinel-2 MSI Level-2A


@dataclass
class Observation:
    """
    Metadata for a single Sentinel-2 observation (product).

    This is NOT an image — it is a record describing one satellite pass
    that produced imagery for the given area.
    """

    product_id: str
    """Unique product identifier in CDSE (UUID)."""

    name: str
    """Human-readable product name (e.g. S2A_MSIL2A_20261001...)."""

    acquisition_date: date
    """UTC date of the satellite acquisition."""

    acquisition_datetime: datetime
    """Full UTC datetime of the satellite acquisition."""

    platform: str
    """Satellite platform, e.g. 'Sentinel-2A' or 'Sentinel-2B'."""

    cloud_coverage: float
    """Estimated cloud coverage percentage (0–100)."""

    bbox: BoundingBox
    """Geographic bounding box of the product footprint."""

    online: bool = True
    """Whether the product is immediately available for download."""

    download_url: Optional[str] = None
    """Direct download URL if available."""

    size_mb: Optional[float] = None
    """Product size in MB."""

    extra: dict = field(default_factory=dict)
    """Additional metadata from the API response."""

    def __str__(self) -> str:
        return (
            f"Observation({self.name[:30]}… | "
            f"date={self.acquisition_date} | "
            f"cloud={self.cloud_coverage:.1f}% | "
            f"online={self.online})"
        )


class CDSECatalog:
    """
    Searches the Copernicus Data Space Ecosystem catalog for
    Sentinel-2 L2A products.

    Uses the OData v1 REST API which supports filtering by
    collection, date range, area of interest, and cloud coverage.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._auth = get_authenticator()
        self._session = requests.Session()
        self._session.timeout = 60

    def search(
        self,
        bbox: BoundingBox,
        start_date: date,
        end_date: date,
        max_cloud_coverage: float = 20.0,
        max_results: int = 50,
    ) -> List[Observation]:
        """
        Search for Sentinel-2 L2A observations within a bounding box and
        date range.

        Args:
            bbox:               Geographic bounding box.
            start_date:         Earliest acquisition date (inclusive).
            end_date:           Latest acquisition date (inclusive).
            max_cloud_coverage: Maximum allowed cloud coverage (0–100).
            max_results:        Maximum number of results to return.

        Returns:
            List of Observation objects sorted by cloud coverage ascending.

        Raises:
            RuntimeError: If credentials are missing or the API returns an error.
        """
        logger.info(
            "Searching Sentinel-2 L2A observations | bbox=%s | %s → %s | cloud≤%.0f%%",
            bbox,
            start_date,
            end_date,
            max_cloud_coverage,
        )

        self._settings.require_copernicus_credentials()
        headers = self._auth.get_auth_headers()

        # Build OData filter expression
        filter_parts = [
            f"Collection/Name eq '{_S2_L2A_COLLECTION}'",
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
            f"and att/OData.CSC.StringAttribute/Value eq '{_S2_L2A_PRODUCT_TYPE}')",
            f"ContentDate/Start gt {start_date.strftime('%Y-%m-%d')}T00:00:00.000Z",
            f"ContentDate/Start lt {(end_date + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00.000Z",
            (
                f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
                f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud_coverage})"
            ),
            (
                f"OData.CSC.Intersects(area=geography'SRID=4326;{bbox.to_wkt()}')"
            ),
        ]

        filter_str = " and ".join(filter_parts)

        # Build URL manually to preserve OData special characters
        # (requests' params encoding double-encodes single quotes in OData)
        query_string = (
            f"$filter={filter_str}"
            f"&$orderby=ContentDate/Start desc"
            f"&$top={max_results}"
            f"&$expand=Attributes"
        )
        # Encode only the characters that must be percent-encoded in URLs,
        # but preserve OData syntax characters
        from urllib.parse import quote
        encoded_qs = quote(query_string, safe="$&=,()./ ':;@!*_-+~?#[]")
        url = f"{_ODATA_PRODUCTS_URL}?{encoded_qs}"

        logger.debug("CDSE OData request URL (first 200 chars): %s...", url[:200])

        try:
            response = self._session.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Cannot connect to the Copernicus catalog. "
                "Check your network connection."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            raise RuntimeError(
                f"CDSE catalog search failed (HTTP {status}). "
                "Check your credentials and the date/bbox parameters."
            ) from exc

        data = response.json()
        raw_products = data.get("value", [])

        logger.info("Found %d observations.", len(raw_products))

        observations = []
        for prod in raw_products:
            obs = self._parse_product(prod)
            if obs is not None:
                observations.append(obs)

        # Sort by cloud coverage (best quality first)
        observations.sort(key=lambda o: o.cloud_coverage)
        return observations

    def search_around_date(
        self,
        bbox: BoundingBox,
        target_date: date,
        window_days: int = 15,
        max_cloud_coverage: float = 20.0,
    ) -> List[Observation]:
        """
        Search for observations in a symmetric window around a target date.

        Args:
            bbox:               Geographic bounding box.
            target_date:        Central date for the search window.
            window_days:        Number of days either side of target_date.
            max_cloud_coverage: Maximum cloud coverage (0–100).

        Returns:
            List of Observation objects sorted by temporal distance from
            target_date (closest first), then by cloud coverage.
        """
        start = target_date - timedelta(days=window_days)
        end = target_date + timedelta(days=window_days)

        obs = self.search(
            bbox=bbox,
            start_date=start,
            end_date=end,
            max_cloud_coverage=max_cloud_coverage,
        )

        # Secondary sort: closest to target_date first
        obs.sort(key=lambda o: abs((o.acquisition_date - target_date).days))
        return obs

    def get_best_observation(
        self,
        bbox: BoundingBox,
        target_date: date,
        window_days: int = 15,
        max_cloud_coverage: float = 20.0,
        strategy: str = "nearest",
    ) -> Optional[Observation]:
        """
        Find the single best observation for a target date.

        Args:
            bbox:               Geographic bounding box.
            target_date:        Target date.
            window_days:        Search window (±days).
            max_cloud_coverage: Maximum cloud coverage.
            strategy:           'nearest' (closest date) or
                                'clearest' (lowest cloud coverage).

        Returns:
            Best matching Observation, or None if nothing found.
        """
        candidates = self.search_around_date(
            bbox=bbox,
            target_date=target_date,
            window_days=window_days,
            max_cloud_coverage=max_cloud_coverage,
        )

        if not candidates:
            logger.warning(
                "No observations found within ±%d days of %s with cloud≤%.0f%%",
                window_days,
                target_date,
                max_cloud_coverage,
            )
            return None

        if strategy == "nearest":
            # Already sorted by temporal distance
            best = candidates[0]
        elif strategy == "clearest":
            best = min(candidates, key=lambda o: o.cloud_coverage)
        else:
            raise ValueError(f"Unknown strategy {strategy!r}. Use 'nearest' or 'clearest'.")

        logger.info(
            "Selected observation: %s (cloud=%.1f%%)",
            best.name,
            best.cloud_coverage,
        )
        return best

    def _parse_product(self, prod: dict) -> Optional[Observation]:
        """
        Parse a raw OData product dict into an Observation object.

        Args:
            prod: Raw dict from the CDSE OData API.

        Returns:
            Observation instance, or None if the product is invalid.
        """
        try:
            product_id = prod["Id"]
            name = prod["Name"]

            # Parse acquisition datetime
            raw_dt = prod.get("ContentDate", {}).get("Start", "")
            if raw_dt:
                acq_dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            else:
                logger.warning("Product %s has no ContentDate, skipping.", product_id)
                return None

            # Extract cloud coverage from extended attributes
            cloud_coverage = 100.0
            for attr in prod.get("Attributes", []):
                if attr.get("Name") == "cloudCover":
                    try:
                        cloud_coverage = float(attr["Value"])
                    except (TypeError, ValueError):
                        pass

            # Extract platform name
            platform = prod.get("Satellite", "Sentinel-2")

            # Extract download URL and online status
            online = prod.get("Online", True)

            # Parse footprint / geometry as bounding box
            # CDSE returns a WKT or GeoJSON footprint
            footprint = prod.get("Footprint", "")
            bbox = self._parse_footprint_bbox(footprint, product_id)

            # File size
            size_bytes = prod.get("ContentLength", 0)
            size_mb = size_bytes / (1024 * 1024) if size_bytes else None

            return Observation(
                product_id=product_id,
                name=name,
                acquisition_date=acq_dt.date(),
                acquisition_datetime=acq_dt,
                platform=platform,
                cloud_coverage=cloud_coverage,
                bbox=bbox,
                online=online,
                size_mb=size_mb,
                extra=prod,
            )

        except Exception as exc:
            logger.warning("Failed to parse product record: %s", exc)
            return None

    @staticmethod
    def _parse_footprint_bbox(footprint: str, product_id: str) -> BoundingBox:
        """
        Extract a BoundingBox from a WKT POLYGON footprint string.
        Falls back to a global bbox if parsing fails.
        """
        try:
            if footprint and "POLYGON" in footprint.upper():
                from shapely import wkt as shapely_wkt
                geom = shapely_wkt.loads(footprint)
                b = geom.bounds  # (minx, miny, maxx, maxy)
                return BoundingBox(west=b[0], south=b[1], east=b[2], north=b[3])
        except Exception as exc:
            logger.debug("Could not parse footprint for %s: %s", product_id, exc)

        # Fallback: very wide bbox (signals parsing failure)
        return BoundingBox(west=-180.0, south=-90.0, east=180.0, north=90.0)
