"""
Satellite provider abstraction layer.

Defines the SatelliteProvider abstract base class and the
Sentinel2Provider concrete implementation, which composes the
catalog search and downloader modules into a single interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

from backend.app.satellite.catalog import CDSECatalog, Observation
from backend.app.satellite.downloader import DEFAULT_BANDS, DownloadedImage, Sentinel2Downloader
from backend.app.utils.config import get_settings
from backend.app.utils.geo import BoundingBox
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class SatelliteProvider(ABC):
    """
    Abstract interface for satellite imagery providers.

    Concrete implementations must implement all abstract methods.
    This abstraction allows swapping providers (e.g., Planet, Landsat)
    without changing downstream code.
    """

    @abstractmethod
    def search_observations(
        self,
        bbox: BoundingBox,
        start_date: date,
        end_date: date,
        max_cloud_coverage: float = 20.0,
    ) -> List[Observation]:
        """Search for available satellite observations."""
        ...

    @abstractmethod
    def get_best_observation(
        self,
        bbox: BoundingBox,
        target_date: date,
        window_days: int = 15,
        max_cloud_coverage: float = 20.0,
        strategy: str = "nearest",
    ) -> Optional[Observation]:
        """Return the single best observation for a target date."""
        ...

    @abstractmethod
    def download_image(
        self,
        observation: Observation,
        bbox: BoundingBox,
        bands: Optional[List[str]] = None,
        resolution_m: int = 10,
        force_download: bool = False,
    ) -> DownloadedImage:
        """Download imagery for an observation."""
        ...

    @abstractmethod
    def get_metadata(self, observation: Observation) -> dict:
        """Return metadata dict for an observation."""
        ...


class Sentinel2Provider(SatelliteProvider):
    """
    Sentinel-2 L2A satellite imagery provider.

    Uses:
    - CDSECatalog for searching available observations via the
      Copernicus Data Space Ecosystem OData API.
    - Sentinel2Downloader for downloading imagery via the
      Sentinel Hub Process API (with CDSE S3 fallback).
    """

    def __init__(self) -> None:
        self._catalog = CDSECatalog()
        self._downloader = Sentinel2Downloader()
        self._settings = get_settings()

    def search_observations(
        self,
        bbox: BoundingBox,
        start_date: date,
        end_date: date,
        max_cloud_coverage: float = 20.0,
    ) -> List[Observation]:
        """
        Search for Sentinel-2 L2A observations.

        Args:
            bbox:               Area of interest.
            start_date:         Start of search window.
            end_date:           End of search window.
            max_cloud_coverage: Maximum cloud coverage (0–100).

        Returns:
            List of matching Observation objects, sorted by cloud coverage.
        """
        return self._catalog.search(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_coverage=max_cloud_coverage,
        )

    def get_best_observation(
        self,
        bbox: BoundingBox,
        target_date: date,
        window_days: int = 15,
        max_cloud_coverage: float = 20.0,
        strategy: str = "nearest",
    ) -> Optional[Observation]:
        """
        Find the single best Sentinel-2 observation.

        Args:
            bbox:               Area of interest.
            target_date:        Target acquisition date.
            window_days:        Search window (±days around target_date).
            max_cloud_coverage: Maximum allowed cloud coverage.
            strategy:           'nearest' (by date) or 'clearest' (by cloud).

        Returns:
            Best Observation, or None if nothing found.
        """
        return self._catalog.get_best_observation(
            bbox=bbox,
            target_date=target_date,
            window_days=window_days,
            max_cloud_coverage=max_cloud_coverage,
            strategy=strategy,
        )

    def download_image(
        self,
        observation: Observation,
        bbox: BoundingBox,
        bands: Optional[List[str]] = None,
        resolution_m: int = 10,
        force_download: bool = False,
    ) -> DownloadedImage:
        """
        Download Sentinel-2 imagery for an observation.

        Args:
            observation:    The observation to download.
            bbox:           Area of interest.
            bands:          Band names (defaults to DEFAULT_BANDS).
            resolution_m:   Output pixel size in metres (10, 20, 60).
            force_download: Ignore cache and re-download.

        Returns:
            DownloadedImage with local path and metadata.
        """
        if bands is None:
            bands = list(DEFAULT_BANDS)

        return self._downloader.download(
            observation=observation,
            bbox=bbox,
            bands=bands,
            resolution_m=resolution_m,
            force_download=force_download,
        )

    def get_metadata(self, observation: Observation) -> dict:
        """
        Return a metadata dict for an observation.

        Args:
            observation: Target observation.

        Returns:
            Dict with acquisition metadata suitable for JSON serialization.
        """
        return {
            "product_id": observation.product_id,
            "name": observation.name,
            "acquisition_date": observation.acquisition_date.isoformat(),
            "acquisition_datetime": observation.acquisition_datetime.isoformat(),
            "platform": observation.platform,
            "cloud_coverage": observation.cloud_coverage,
            "bbox": {
                "west": observation.bbox.west,
                "south": observation.bbox.south,
                "east": observation.bbox.east,
                "north": observation.bbox.north,
            },
            "online": observation.online,
            "size_mb": observation.size_mb,
        }

    def find_image_pair(
        self,
        bbox: BoundingBox,
        before_date: date,
        after_date: date,
        window_days: int = 15,
        max_cloud_coverage: float = 20.0,
        strategy: str = "nearest",
    ) -> tuple[Optional[Observation], Optional[Observation]]:
        """
        Find a before/after image pair for change detection.

        Each date is searched independently with its own window.
        The same observation will not be returned for both slots.

        Args:
            bbox:               Area of interest.
            before_date:        Target date for the 'before' image.
            after_date:         Target date for the 'after' image.
            window_days:        Search window (±days) for each date.
            max_cloud_coverage: Maximum cloud coverage.
            strategy:           'nearest' or 'clearest'.

        Returns:
            Tuple of (before_observation, after_observation).
            Either may be None if no suitable observation is found.
        """
        logger.info(
            "Searching for image pair | before≈%s | after≈%s | cloud≤%.0f%%",
            before_date,
            after_date,
            max_cloud_coverage,
        )

        before_obs = self.get_best_observation(
            bbox=bbox,
            target_date=before_date,
            window_days=window_days,
            max_cloud_coverage=max_cloud_coverage,
            strategy=strategy,
        )

        after_obs = self.get_best_observation(
            bbox=bbox,
            target_date=after_date,
            window_days=window_days,
            max_cloud_coverage=max_cloud_coverage,
            strategy=strategy,
        )

        # Ensure they are different observations
        if (
            before_obs
            and after_obs
            and before_obs.product_id == after_obs.product_id
        ):
            logger.warning(
                "Before and after resolved to the same observation (%s). "
                "Try a larger date gap or wider search window.",
                before_obs.name,
            )
            after_obs = None

        return before_obs, after_obs
