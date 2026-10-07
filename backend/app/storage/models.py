"""
SQLAlchemy ORM models for the change detection system.

Tables:
- Observation: A Sentinel-2 product record from the catalog.
- AnalysisPair: A before/after image pair selected for comparison.
- AnalysisResult: The output of a change detection analysis.
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.app.storage.database import Base


class ObservationRecord(Base):
    """Stores metadata for a Sentinel-2 observation (catalog record)."""

    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(256), nullable=False)
    acquisition_date = Column(String(10), nullable=False, index=True)
    acquisition_datetime = Column(String(32), nullable=False)
    platform = Column(String(32), default="Sentinel-2")
    cloud_coverage = Column(Float, default=0.0)

    # Bounding box
    bbox_west = Column(Float)
    bbox_south = Column(Float)
    bbox_east = Column(Float)
    bbox_north = Column(Float)

    # Download status
    online = Column(Boolean, default=True)
    downloaded = Column(Boolean, default=False)
    local_path = Column(String(512), nullable=True)
    bands = Column(Text, default="[]")  # JSON list

    size_mb = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    as_before = relationship("AnalysisPairRecord", foreign_keys="AnalysisPairRecord.before_id", back_populates="before_obs")
    as_after = relationship("AnalysisPairRecord", foreign_keys="AnalysisPairRecord.after_id", back_populates="after_obs")

    def get_bands(self) -> list[str]:
        """Return the band list as Python list."""
        try:
            return json.loads(self.bands)
        except Exception:
            return []

    def set_bands(self, band_list: list[str]) -> None:
        """Store band list as JSON string."""
        self.bands = json.dumps(band_list)

    def __repr__(self) -> str:
        return f"<Observation {self.name[:30]} | {self.acquisition_date} | cloud={self.cloud_coverage:.1f}%>"


class AnalysisPairRecord(Base):
    """Stores a before/after image pair selected for change detection."""

    __tablename__ = "analysis_pairs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    before_id = Column(Integer, ForeignKey("observations.id"), nullable=False)
    after_id = Column(Integer, ForeignKey("observations.id"), nullable=False)

    before_date = Column(String(10), nullable=False)
    after_date = Column(String(10), nullable=False)
    temporal_gap_days = Column(Integer, default=0)

    # AOI (may differ from observation footprints)
    aoi_west = Column(Float)
    aoi_south = Column(Float)
    aoi_east = Column(Float)
    aoi_north = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    before_obs = relationship("ObservationRecord", foreign_keys=[before_id], back_populates="as_before")
    after_obs = relationship("ObservationRecord", foreign_keys=[after_id], back_populates="as_after")
    results = relationship("AnalysisResultRecord", back_populates="pair")

    def __repr__(self) -> str:
        return f"<Pair {self.before_date} → {self.after_date}>"


class AnalysisResultRecord(Base):
    """Stores the results of a completed change detection analysis."""

    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey("analysis_pairs.id"), nullable=False)

    # Status
    status = Column(String(32), default="pending")  # pending, running, complete, error
    error_message = Column(Text, nullable=True)

    # Results
    change_percentage = Column(Float, nullable=True)
    changed_area_km2 = Column(Float, nullable=True)
    valid_area_km2 = Column(Float, nullable=True)
    total_area_km2 = Column(Float, nullable=True)
    n_changed_pixels = Column(Integer, nullable=True)
    n_valid_pixels = Column(Integer, nullable=True)

    # Index-based hints (JSON)
    index_hints = Column(Text, default="{}")

    # Output file paths (JSON dict)
    output_paths = Column(Text, default="{}")

    # Reproducibility metadata (JSON)
    analysis_config = Column(Text, default="{}")

    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    pair = relationship("AnalysisPairRecord", back_populates="results")

    def get_index_hints(self) -> dict:
        try:
            return json.loads(self.index_hints)
        except Exception:
            return {}

    def get_output_paths(self) -> dict:
        try:
            return json.loads(self.output_paths)
        except Exception:
            return {}

    def get_analysis_config(self) -> dict:
        try:
            return json.loads(self.analysis_config)
        except Exception:
            return {}

    def __repr__(self) -> str:
        return f"<Result pair={self.pair_id} status={self.status} change={self.change_percentage}%>"
