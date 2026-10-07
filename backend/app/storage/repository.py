"""
Repository pattern for database CRUD operations.

Provides clean, testable data access methods that keep
SQL/ORM details out of business logic.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from backend.app.storage.models import AnalysisPairRecord, AnalysisResultRecord, ObservationRecord
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


class ObservationRepository:
    """Repository for Observation records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_product_id(self, product_id: str) -> Optional[ObservationRecord]:
        """Return observation by Copernicus product ID."""
        return (
            self._session.query(ObservationRecord)
            .filter(ObservationRecord.product_id == product_id)
            .first()
        )

    def create_or_update(self, **kwargs) -> ObservationRecord:
        """Create or update an observation record."""
        product_id = kwargs.get("product_id")
        existing = self.get_by_product_id(product_id)
        if existing:
            for k, v in kwargs.items():
                setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
            self._session.commit()
            return existing

        record = ObservationRecord(**kwargs)
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def list_by_date_range(
        self,
        start: date,
        end: date,
    ) -> List[ObservationRecord]:
        """Return all observations within a date range."""
        return (
            self._session.query(ObservationRecord)
            .filter(
                ObservationRecord.acquisition_date >= start.isoformat(),
                ObservationRecord.acquisition_date <= end.isoformat(),
            )
            .order_by(ObservationRecord.acquisition_date.desc())
            .all()
        )

    def list_downloaded(self) -> List[ObservationRecord]:
        """Return all observations that have been downloaded."""
        return (
            self._session.query(ObservationRecord)
            .filter(ObservationRecord.downloaded.is_(True))
            .all()
        )


class AnalysisPairRepository:
    """Repository for AnalysisPair records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        before_id: int,
        after_id: int,
        before_date: str,
        after_date: str,
        aoi_bbox: tuple,
    ) -> AnalysisPairRecord:
        """Create a new analysis pair."""
        from datetime import date as dt_date
        try:
            d1 = date.fromisoformat(before_date)
            d2 = date.fromisoformat(after_date)
            gap = abs((d2 - d1).days)
        except Exception:
            gap = 0

        record = AnalysisPairRecord(
            before_id=before_id,
            after_id=after_id,
            before_date=before_date,
            after_date=after_date,
            temporal_gap_days=gap,
            aoi_west=aoi_bbox[0],
            aoi_south=aoi_bbox[1],
            aoi_east=aoi_bbox[2],
            aoi_north=aoi_bbox[3],
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_by_id(self, pair_id: int) -> Optional[AnalysisPairRecord]:
        """Return a pair by ID."""
        return self._session.query(AnalysisPairRecord).filter_by(id=pair_id).first()

    def list_all(self) -> List[AnalysisPairRecord]:
        """Return all analysis pairs ordered by creation date."""
        return (
            self._session.query(AnalysisPairRecord)
            .order_by(AnalysisPairRecord.created_at.desc())
            .all()
        )


class AnalysisResultRepository:
    """Repository for AnalysisResult records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, pair_id: int, config: dict) -> AnalysisResultRecord:
        """Create a new analysis result record in pending state."""
        record = AnalysisResultRecord(
            pair_id=pair_id,
            status="pending",
            analysis_config=json.dumps(config),
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return record

    def get_by_id(self, result_id: int) -> Optional[AnalysisResultRecord]:
        """Return a result by ID."""
        return self._session.query(AnalysisResultRecord).filter_by(id=result_id).first()

    def update_status(self, result_id: int, status: str, error: str = None) -> None:
        """Update the status of a result record."""
        record = self.get_by_id(result_id)
        if record:
            record.status = status
            if error:
                record.error_message = error
            if status == "running":
                record.started_at = datetime.utcnow()
            elif status in ("complete", "error"):
                record.completed_at = datetime.utcnow()
            self._session.commit()

    def update_results(self, result_id: int, stats: dict, paths: dict) -> None:
        """Save computed statistics and output paths."""
        record = self.get_by_id(result_id)
        if record:
            record.change_percentage = stats.get("change_percentage")
            record.changed_area_km2 = stats.get("changed_area_km2")
            record.valid_area_km2 = stats.get("valid_area_km2")
            record.total_area_km2 = stats.get("total_area_km2")
            record.n_changed_pixels = stats.get("n_changed_pixels")
            record.n_valid_pixels = stats.get("n_valid_pixels")
            record.index_hints = json.dumps(stats.get("index_hints", {}))
            record.output_paths = json.dumps(paths)
            record.status = "complete"
            record.completed_at = datetime.utcnow()
            self._session.commit()

    def list_by_pair(self, pair_id: int) -> List[AnalysisResultRecord]:
        """Return all results for a specific pair."""
        return (
            self._session.query(AnalysisResultRecord)
            .filter_by(pair_id=pair_id)
            .order_by(AnalysisResultRecord.created_at.desc())
            .all()
        )
