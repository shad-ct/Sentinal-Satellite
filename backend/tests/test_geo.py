"""Unit tests for geo utilities — bounding box validation and coordinate helpers."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.utils.geo import BoundingBox, validate_bounding_box, validate_coordinates, degrees_to_utm_epsg


class TestValidateCoordinates:
    def test_valid_coordinates(self):
        validate_coordinates(11.87, 75.37)  # Should not raise

    def test_lat_out_of_range(self):
        with pytest.raises(ValueError, match="Latitude"):
            validate_coordinates(91.0, 0.0)

    def test_lon_out_of_range(self):
        with pytest.raises(ValueError, match="Longitude"):
            validate_coordinates(0.0, 181.0)

    def test_negative_valid(self):
        validate_coordinates(-33.87, 151.21)  # Sydney

    def test_poles(self):
        validate_coordinates(90.0, 0.0)
        validate_coordinates(-90.0, 0.0)


class TestBoundingBox:
    def test_from_center(self):
        bbox = BoundingBox.from_center(11.87, 75.37, size_km=10)
        assert bbox.south < 11.87 < bbox.north
        assert bbox.west < 75.37 < bbox.east

    def test_width_height(self):
        bbox = BoundingBox(west=0.0, south=0.0, east=1.0, north=1.0)
        assert abs(bbox.width_deg - 1.0) < 1e-6
        assert abs(bbox.height_deg - 1.0) < 1e-6

    def test_invalid_west_east(self):
        with pytest.raises(ValueError, match="west"):
            BoundingBox(west=10.0, south=0.0, east=5.0, north=1.0)

    def test_invalid_south_north(self):
        with pytest.raises(ValueError, match="south"):
            BoundingBox(west=0.0, south=10.0, east=1.0, north=5.0)

    def test_to_tuple(self):
        bbox = BoundingBox(west=10.0, south=20.0, east=11.0, north=21.0)
        t = bbox.to_tuple()
        assert t == (10.0, 20.0, 11.0, 21.0)

    def test_area_positive(self):
        bbox = BoundingBox.from_center(11.87, 75.37, size_km=10)
        area = bbox.area_km2()
        assert area > 0

    def test_wkt_polygon(self):
        bbox = BoundingBox(west=0.0, south=0.0, east=1.0, north=1.0)
        wkt = bbox.to_wkt()
        assert "POLYGON" in wkt

    def test_geojson(self):
        bbox = BoundingBox(west=0.0, south=0.0, east=1.0, north=1.0)
        gj = bbox.to_geojson()
        assert gj["type"] == "Polygon"


class TestUTMZone:
    def test_northern_hemisphere(self):
        epsg = degrees_to_utm_epsg(11.87, 75.37)  # India
        assert 32600 <= epsg <= 32660

    def test_southern_hemisphere(self):
        epsg = degrees_to_utm_epsg(-33.87, 151.21)  # Sydney
        assert 32700 <= epsg <= 32760
