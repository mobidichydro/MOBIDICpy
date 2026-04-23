"""Tests for mobidic.core.crop_coefficients module."""

from pathlib import Path

import numpy as np
import pytest

from mobidic.core.crop_coefficients import (
    _MONTH_COLUMNS,
    compute_kc_grid,
    default_kc_clc_mapping_path,
    load_kc_clc_mapping,
)


class TestDefaultKcClcMappingPath:
    """Tests for default_kc_clc_mapping_path."""

    def test_default_path_exists(self):
        """Default mapping file shipped with the package must exist."""
        p = default_kc_clc_mapping_path()
        assert isinstance(p, Path)
        assert p.exists()
        assert p.suffix == ".csv"


class TestLoadKcClcMapping:
    """Tests for load_kc_clc_mapping."""

    def test_load_default_mapping(self):
        """Loading the default mapping returns a non-empty dict of 12-valued arrays."""
        mapping = load_kc_clc_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        for code, values in mapping.items():
            assert isinstance(code, int)
            assert isinstance(values, np.ndarray)
            assert values.shape == (12,)
            assert values.dtype == np.float64

    def test_load_custom_mapping(self, tmp_path):
        """A custom CSV file is parsed correctly, including comment lines."""
        csv = tmp_path / "custom_kc.csv"
        csv.write_text(
            "# comment line to be ignored\n"
            "clc_code," + ",".join(_MONTH_COLUMNS) + "\n"
            "111,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.45,0.35,0.25,0.15,0.10\n"
            "211,0.30,0.35,0.60,0.90,1.10,1.15,1.10,0.90,0.70,0.50,0.40,0.30\n"
        )
        mapping = load_kc_clc_mapping(csv)
        assert set(mapping.keys()) == {111, 211}
        np.testing.assert_allclose(
            mapping[111],
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.45, 0.35, 0.25, 0.15, 0.10],
        )
        np.testing.assert_allclose(mapping[211][0], 0.30)
        np.testing.assert_allclose(mapping[211][5], 1.15)

    def test_file_not_found(self, tmp_path):
        """A non-existent path raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError, match="Kc/CLC mapping file not found"):
            load_kc_clc_mapping(missing)

    def test_missing_columns_raises(self, tmp_path):
        """A CSV that is missing some required columns raises ValueError."""
        csv = tmp_path / "bad.csv"
        # Omit kc_dec
        csv.write_text("clc_code," + ",".join(_MONTH_COLUMNS[:-1]) + "\n" + "111," + ",".join(["0.1"] * 11) + "\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_kc_clc_mapping(csv)

    def test_accepts_string_path(self, tmp_path):
        """A string path (not Path object) is accepted."""
        csv = tmp_path / "ok.csv"
        csv.write_text("clc_code," + ",".join(_MONTH_COLUMNS) + "\n" + "111," + ",".join(["0.2"] * 12) + "\n")
        mapping = load_kc_clc_mapping(str(csv))
        assert 111 in mapping


class TestComputeKcGrid:
    """Tests for compute_kc_grid."""

    @pytest.fixture
    def mapping(self):
        """Simple mapping with two CLC classes and known month values."""
        return {
            111: np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.45, 0.35, 0.25, 0.15, 0.10]),
            211: np.array([0.30, 0.35, 0.60, 0.90, 1.10, 1.15, 1.10, 0.90, 0.70, 0.50, 0.40, 0.30]),
        }

    def test_invalid_month_raises(self, mapping):
        """Month outside 1..12 raises ValueError."""
        grid = np.array([[111.0]])
        with pytest.raises(ValueError, match="month must be in 1..12"):
            compute_kc_grid(grid, mapping, month=0, default_kc=1.0)
        with pytest.raises(ValueError, match="month must be in 1..12"):
            compute_kc_grid(grid, mapping, month=13, default_kc=1.0)

    def test_none_grid_returns_scalar_default(self, mapping):
        """When clc_grid is None the scalar default_kc is returned."""
        result = compute_kc_grid(None, mapping, month=6, default_kc=0.85)
        assert isinstance(result, float)
        assert result == pytest.approx(0.85)

    def test_known_codes_mapped_per_month(self, mapping):
        """Cells with known CLC codes pick up the correct monthly Kc value."""
        grid = np.array(
            [
                [111.0, 211.0],
                [211.0, 111.0],
            ]
        )
        # July -> index 6
        kc = compute_kc_grid(grid, mapping, month=7, default_kc=0.99)
        assert isinstance(kc, np.ndarray)
        assert kc.shape == grid.shape
        np.testing.assert_allclose(kc[0, 0], 0.50)
        np.testing.assert_allclose(kc[0, 1], 1.10)
        np.testing.assert_allclose(kc[1, 0], 1.10)
        np.testing.assert_allclose(kc[1, 1], 0.50)

    def test_unknown_codes_use_default(self, mapping):
        """Cells with codes absent from the mapping fall back to default_kc."""
        grid = np.array([[111.0, 999.0], [888.0, 211.0]])
        kc = compute_kc_grid(grid, mapping, month=1, default_kc=0.77)
        np.testing.assert_allclose(kc[0, 0], 0.10)  # 111 jan
        np.testing.assert_allclose(kc[0, 1], 0.77)  # unknown -> default
        np.testing.assert_allclose(kc[1, 0], 0.77)  # unknown -> default
        np.testing.assert_allclose(kc[1, 1], 0.30)  # 211 jan

    def test_nan_cells_use_default(self, mapping):
        """NaN cells in the CLC grid stay at default_kc."""
        grid = np.array([[111.0, np.nan], [np.nan, 211.0]])
        kc = compute_kc_grid(grid, mapping, month=6, default_kc=0.5)
        np.testing.assert_allclose(kc[0, 0], 0.40)
        np.testing.assert_allclose(kc[0, 1], 0.5)
        np.testing.assert_allclose(kc[1, 0], 0.5)
        np.testing.assert_allclose(kc[1, 1], 1.15)

    def test_all_nan_grid_returns_default_grid(self, mapping):
        """A grid of all NaN values returns a grid filled with default_kc."""
        grid = np.full((3, 3), np.nan)
        kc = compute_kc_grid(grid, mapping, month=5, default_kc=0.42)
        assert isinstance(kc, np.ndarray)
        assert kc.shape == (3, 3)
        np.testing.assert_allclose(kc, 0.42)

    def test_non_integer_codes_rounded(self, mapping):
        """Non-integer CLC codes (from float raster) are rounded to the nearest int."""
        grid = np.array([[111.2, 210.6]])  # 111.2 -> 111 ; 210.6 -> 211
        kc = compute_kc_grid(grid, mapping, month=3, default_kc=0.0)
        np.testing.assert_allclose(kc[0, 0], 0.20)
        np.testing.assert_allclose(kc[0, 1], 0.60)
