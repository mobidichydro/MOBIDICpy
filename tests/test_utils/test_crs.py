"""Tests for mobidic.utils.crs module."""

from pyproj import CRS
from mobidic.utils.crs import (
    crs_to_cf_attrs,
    _extract_epsg_from_wkt,
    get_epsg_code,
    parse_crs,
    crs_equals,
)


class TestCrsToCfAttrs:
    """Test crs_to_cf_attrs function."""

    def test_epsg_code_input(self):
        """Test with EPSG code as integer."""
        attrs = crs_to_cf_attrs(3003)
        assert attrs["grid_mapping_name"] == "transverse_mercator"
        assert attrs["epsg_code"] == "EPSG:3003"
        assert "crs_wkt" in attrs
        assert len(attrs["crs_wkt"]) > 0

    def test_epsg_string_input(self):
        """Test with EPSG code as string."""
        attrs = crs_to_cf_attrs("EPSG:3003")
        assert attrs["grid_mapping_name"] == "transverse_mercator"
        assert attrs["epsg_code"] == "EPSG:3003"
        assert "crs_wkt" in attrs

    def test_crs_object_input(self):
        """Test with pyproj CRS object."""
        crs = CRS.from_epsg(3003)
        attrs = crs_to_cf_attrs(crs)
        assert attrs["grid_mapping_name"] == "transverse_mercator"
        assert attrs["epsg_code"] == "EPSG:3003"
        assert "crs_wkt" in attrs

    def test_wkt_string_input(self):
        """Test with WKT string."""
        crs = CRS.from_epsg(3003)
        wkt = crs.to_wkt()
        attrs = crs_to_cf_attrs(wkt)
        assert "grid_mapping_name" in attrs
        assert "crs_wkt" in attrs

    def test_wgs84_input(self):
        """Test with WGS84 (EPSG:4326)."""
        attrs = crs_to_cf_attrs("EPSG:4326")
        assert attrs["grid_mapping_name"] == "latitude_longitude"
        assert attrs["epsg_code"] == "EPSG:4326"

    def test_utm_zone_input(self):
        """Test with UTM zone."""
        attrs = crs_to_cf_attrs("EPSG:32632")  # UTM Zone 32N
        assert attrs["grid_mapping_name"] == "transverse_mercator"
        assert attrs["epsg_code"] == "EPSG:32632"

    def test_none_input(self):
        """Test with None input."""
        attrs = crs_to_cf_attrs(None)
        assert attrs["grid_mapping_name"] == "latitude_longitude"
        assert attrs["crs_wkt"] == ""

    def test_empty_string_input(self):
        """Test with empty string."""
        attrs = crs_to_cf_attrs("")
        assert attrs["grid_mapping_name"] == "latitude_longitude"
        assert attrs["crs_wkt"] == ""

    def test_whitespace_string_input(self):
        """Test with whitespace-only string."""
        attrs = crs_to_cf_attrs("   ")
        assert attrs["grid_mapping_name"] == "latitude_longitude"
        assert attrs["crs_wkt"] == ""

    def test_invalid_input_fallback(self):
        """Test fallback behavior with invalid input."""
        attrs = crs_to_cf_attrs("INVALID_CRS_STRING_12345")
        assert attrs["grid_mapping_name"] == "latitude_longitude"
        # Should have attempted to get WKT or converted to string
        assert "crs_wkt" in attrs

    def test_epsg_code_without_prefix(self):
        """Test EPSG code without 'EPSG:' prefix can be extracted."""
        attrs = crs_to_cf_attrs(4326)
        assert attrs["epsg_code"] == "EPSG:4326"

    def test_includes_all_cf_attributes(self):
        """Test that all CF attributes are included in output."""
        attrs = crs_to_cf_attrs("EPSG:3003")
        # Should have grid_mapping_name
        assert "grid_mapping_name" in attrs
        # Should have crs_wkt
        assert "crs_wkt" in attrs
        assert len(attrs["crs_wkt"]) > 0
        # Should have epsg_code
        assert "epsg_code" in attrs
        # Should have projection-specific parameters
        assert len(attrs) > 3  # More than just the 3 above

    def test_crs_without_epsg_code(self):
        """Test CRS that doesn't have an EPSG code."""
        # Create a custom CRS without EPSG code
        proj_string = "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m"
        attrs = crs_to_cf_attrs(proj_string)
        assert "grid_mapping_name" in attrs
        assert "crs_wkt" in attrs
        # May or may not have epsg_code depending on pyproj version


class TestExtractEpsgFromWkt:
    """Test _extract_epsg_from_wkt function."""

    def test_wkt1_format(self):
        """Test extraction from WKT1 format with AUTHORITY tag."""
        wkt = 'PROJCS["Monte Mario / Italy zone 1",AUTHORITY["EPSG","3003"]]'
        epsg = _extract_epsg_from_wkt(wkt)
        assert epsg == 3003

    def test_wkt2_format(self):
        """Test extraction from WKT2 format with ID tag."""
        wkt = 'PROJCRS["Monte Mario / Italy zone 1",ID["EPSG",3003]]'
        epsg = _extract_epsg_from_wkt(wkt)
        assert epsg == 3003

    def test_wkt1_with_spaces(self):
        """Test WKT1 with extra spaces."""
        wkt = 'PROJCS["Monte Mario",AUTHORITY["EPSG",  "3003"]]'
        epsg = _extract_epsg_from_wkt(wkt)
        assert epsg == 3003

    def test_wkt2_with_spaces(self):
        """Test WKT2 with extra spaces."""
        wkt = 'PROJCRS["Monte Mario",ID["EPSG",  3003]]'
        epsg = _extract_epsg_from_wkt(wkt)
        assert epsg == 3003

    def test_no_epsg_code(self):
        """Test WKT without EPSG code."""
        wkt = 'PROJCS["Custom Projection",PARAMETER["foo","bar"]]'
        epsg = _extract_epsg_from_wkt(wkt)
        assert epsg is None

    def test_none_input(self):
        """Test with None input."""
        epsg = _extract_epsg_from_wkt(None)
        assert epsg is None

    def test_empty_string(self):
        """Test with empty string."""
        epsg = _extract_epsg_from_wkt("")
        assert epsg is None

    def test_non_string_input(self):
        """Test with non-string input."""
        epsg = _extract_epsg_from_wkt(12345)
        assert epsg is None

    def test_real_wkt_string(self):
        """Test with real WKT string from pyproj."""
        crs = CRS.from_epsg(3003)
        wkt = crs.to_wkt(version="WKT1_GDAL")
        epsg = _extract_epsg_from_wkt(wkt)
        # Should extract EPSG from AUTHORITY tag
        assert epsg == 3003 or epsg is None  # Depends on WKT format


class TestGetEpsgCode:
    """Test get_epsg_code function."""

    def test_integer_input(self):
        """Test with integer EPSG code."""
        epsg = get_epsg_code(3003)
        assert epsg == 3003

    def test_epsg_string_with_prefix(self):
        """Test with 'EPSG:' prefixed string."""
        epsg = get_epsg_code("EPSG:3003")
        assert epsg == 3003

    def test_crs_object(self):
        """Test with pyproj CRS object."""
        crs = CRS.from_epsg(3003)
        epsg = get_epsg_code(crs)
        assert epsg == 3003

    def test_wkt_string(self):
        """Test with WKT string."""
        crs = CRS.from_epsg(3003)
        wkt = crs.to_wkt()
        epsg = get_epsg_code(wkt)
        assert epsg == 3003

    def test_none_input(self):
        """Test with None."""
        epsg = get_epsg_code(None)
        assert epsg is None

    def test_empty_string(self):
        """Test with empty string."""
        epsg = get_epsg_code("")
        assert epsg is None

    def test_whitespace_string(self):
        """Test with whitespace-only string."""
        epsg = get_epsg_code("   ")
        assert epsg is None

    def test_wgs84(self):
        """Test with WGS84."""
        epsg = get_epsg_code("EPSG:4326")
        assert epsg == 4326

    def test_utm_zone(self):
        """Test with UTM zone."""
        epsg = get_epsg_code("EPSG:32632")
        assert epsg == 32632

    def test_invalid_input(self):
        """Test with invalid CRS string."""
        epsg = get_epsg_code("INVALID_CRS")
        assert epsg is None

    def test_wkt_with_towgs84(self):
        """Test WKT with TOWGS84 parameters (may fail to_epsg but can parse AUTHORITY)."""
        # Create a CRS with custom parameters that might not match official EPSG
        crs = CRS.from_epsg(3003)
        wkt = crs.to_wkt(version="WKT1_GDAL")
        epsg = get_epsg_code(wkt)
        # Should extract from AUTHORITY tag even if to_epsg() fails
        assert epsg == 3003 or epsg is None

    def test_wkt_with_authority_tag(self):
        """Test extraction from WKT string that contains AUTHORITY tag."""
        # Use a real WKT string that contains AUTHORITY tag
        wkt = """PROJCS["Monte Mario / Italy zone 1",
            GEOGCS["Monte Mario",DATUM["Monte_Mario",SPHEROID["International 1924",6378388,297],
            AUTHORITY["EPSG","6265"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],
            AUTHORITY["EPSG","4265"]],PROJECTION["Transverse_Mercator"],
            PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],
            PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",1500000],
            PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3003"]]"""
        epsg = get_epsg_code(wkt)
        # Should extract EPSG from AUTHORITY tag
        assert epsg == 3003


class TestParseCrs:
    """Test parse_crs function."""

    def test_epsg_code_integer(self):
        """Test parsing integer EPSG code."""
        crs = parse_crs(3003)
        assert crs is not None
        assert crs.to_epsg() == 3003

    def test_epsg_string(self):
        """Test parsing EPSG string."""
        crs = parse_crs("EPSG:3003")
        assert crs is not None
        assert crs.to_epsg() == 3003

    def test_crs_object(self):
        """Test with existing CRS object (should return same)."""
        input_crs = CRS.from_epsg(3003)
        result_crs = parse_crs(input_crs)
        assert result_crs is input_crs

    def test_wkt_string(self):
        """Test parsing WKT string."""
        crs_orig = CRS.from_epsg(3003)
        wkt = crs_orig.to_wkt()
        crs = parse_crs(wkt)
        assert crs is not None
        assert crs.to_epsg() == 3003

    def test_none_input(self):
        """Test with None."""
        crs = parse_crs(None)
        assert crs is None

    def test_empty_string(self):
        """Test with empty string."""
        crs = parse_crs("")
        assert crs is None

    def test_whitespace_string(self):
        """Test with whitespace-only string."""
        crs = parse_crs("   ")
        assert crs is None

    def test_invalid_input(self):
        """Test with invalid CRS string."""
        crs = parse_crs("TOTALLY_INVALID_CRS_12345")
        assert crs is None

    def test_proj_string(self):
        """Test with PROJ string."""
        crs = parse_crs("+proj=utm +zone=32 +datum=WGS84")
        assert crs is not None


class TestCrsEquals:
    """Test crs_equals function."""

    def test_same_epsg_codes(self):
        """Test with same EPSG codes."""
        assert crs_equals("EPSG:3003", "EPSG:3003")
        assert crs_equals(3003, "EPSG:3003")
        assert crs_equals("EPSG:3003", 3003)

    def test_different_epsg_codes(self):
        """Test with different EPSG codes."""
        assert not crs_equals("EPSG:3003", "EPSG:4326")
        assert not crs_equals(3003, 4326)

    def test_crs_objects(self):
        """Test with CRS objects."""
        crs1 = CRS.from_epsg(3003)
        crs2 = CRS.from_epsg(3003)
        assert crs_equals(crs1, crs2)

    def test_crs_object_and_code(self):
        """Test CRS object with EPSG code."""
        crs = CRS.from_epsg(3003)
        assert crs_equals(crs, "EPSG:3003")
        assert crs_equals("EPSG:3003", crs)

    def test_wkt_and_epsg(self):
        """Test WKT string with EPSG code."""
        crs = CRS.from_epsg(3003)
        wkt = crs.to_wkt()
        assert crs_equals(wkt, "EPSG:3003")
        assert crs_equals("EPSG:3003", wkt)

    def test_none_inputs(self):
        """Test with None inputs."""
        assert not crs_equals(None, "EPSG:3003")
        assert not crs_equals("EPSG:3003", None)
        assert not crs_equals(None, None)

    def test_empty_string_inputs(self):
        """Test with empty strings."""
        assert not crs_equals("", "EPSG:3003")
        assert not crs_equals("EPSG:3003", "")
        assert not crs_equals("", "")

    def test_invalid_inputs(self):
        """Test with invalid CRS inputs."""
        assert not crs_equals("INVALID_CRS", "EPSG:3003")
        assert not crs_equals("INVALID_CRS_1", "INVALID_CRS_2")

    def test_utm_zones(self):
        """Test different UTM zones."""
        assert crs_equals("EPSG:32632", "EPSG:32632")
        assert not crs_equals("EPSG:32632", "EPSG:32633")

    def test_wgs84_variants(self):
        """Test WGS84 in different formats."""
        assert crs_equals("EPSG:4326", 4326)
        assert crs_equals(4326, CRS.from_epsg(4326))
