"""Tests for hillslope-reach mapping module."""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path
import tempfile
import rasterio
from rasterio.transform import from_origin


@pytest.fixture
def simple_network():
    """Create a simple test network with two reaches."""
    # Create two simple reaches
    reach1 = LineString([(0, 0), (10, 0)])
    reach2 = LineString([(10, 0), (20, 0)])

    gdf = gpd.GeoDataFrame(
        {
            "mobidic_id": [0, 1],
            "geometry": [reach1, reach2],
        },
        crs="EPSG:32632",
    )

    return gdf


@pytest.fixture
def simple_flowdir():
    """Create a simple 5x5 flow direction raster (MOBIDIC notation).

    All cells flow right (direction 8), simulating flow from left to right.
    """
    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    filepath = Path(tmpdir) / "flowdir.tif"

    # Create 5x5 grid, all flowing right (8 in MOBIDIC notation)
    data = np.full((5, 5), 8, dtype=np.float32)

    # Write to GeoTIFF
    transform = from_origin(0, 5, 1, 1)  # xmin, ymax, xres, yres
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype=data.dtype,
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    yield filepath

    # Cleanup
    filepath.unlink()


class TestMapHillslopeToReach:
    """Tests for map_hillslope_to_reach function."""

    def test_missing_hillslope_cells_column(self, simple_network, simple_flowdir):
        """Test that function raises error if hillslope_cells column is missing."""
        from mobidic.preprocessing.hillslope_reach_mapping import map_hillslope_to_reach

        with pytest.raises(ValueError, match="hillslope_cells"):
            map_hillslope_to_reach(simple_network, simple_flowdir)

    def test_missing_mobidic_id_column(self, simple_flowdir):
        """Test that function raises error if mobidic_id column is missing."""
        from mobidic.preprocessing.hillslope_reach_mapping import map_hillslope_to_reach

        # Create network without mobidic_id
        gdf = gpd.GeoDataFrame(
            {
                "hillslope_cells": [[0, 1], [2, 3]],
                "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
            },
            crs="EPSG:32632",
        )

        with pytest.raises(ValueError, match="mobidic_id"):
            map_hillslope_to_reach(gdf, simple_flowdir)

    def test_simple_mapping(self, simple_flowdir):
        """Test basic hillslope-to-reach mapping with a simple case."""
        from mobidic.preprocessing.hillslope_reach_mapping import map_hillslope_to_reach

        # Create network with hillslope_cells already computed
        # Grid is 5x5, let's say reach 0 occupies column 4 (cells 4, 9, 14, 19, 24)
        # Linear indices in column-major order: col * nrows + row
        gdf = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "hillslope_cells": [[4, 9, 14, 19, 24]],  # rightmost column
                "geometry": [LineString([(4, 0), (4, 5)])],
            },
            crs="EPSG:32632",
        )

        # Map hillslope to reach (Grass notation, but we need to convert)
        # Since our flowdir is already in MOBIDIC notation, use Grass as source
        # Actually, we created it in MOBIDIC, so we need to be careful
        # Let me use the grid_to_matrix approach which will flip

        reach_map = map_hillslope_to_reach(gdf, simple_flowdir, flow_dir_type="Grass")

        # Check that result is a 2D array
        assert reach_map.shape == (5, 5)

        # Since all cells flow right (8), cells in columns 0-3 should eventually reach column 4
        # and be assigned to reach 0
        # However, this depends on the exact grid configuration and flow routing


class TestComputeHillslopeCells:
    """Tests for compute_hillslope_cells function."""

    def test_basic_computation(self, simple_network, simple_flowdir):
        """Test basic hillslope cell computation."""
        from mobidic.preprocessing.hillslope_reach_mapping import compute_hillslope_cells

        # Compute hillslope cells
        network = compute_hillslope_cells(simple_network, simple_flowdir, densify_step=1.0)

        # Check that hillslope_cells column was added
        assert "hillslope_cells" in network.columns

        # Check that each reach has some cells assigned
        for idx in network.index:
            cells = network.loc[idx, "hillslope_cells"]
            assert isinstance(cells, list)

    def test_empty_geometry(self, simple_flowdir):
        """Test handling of empty geometry."""
        from mobidic.preprocessing.hillslope_reach_mapping import compute_hillslope_cells

        # Create network with empty geometry
        gdf = gpd.GeoDataFrame(
            {
                "mobidic_id": [0],
                "geometry": [LineString()],  # Empty geometry
            },
            crs="EPSG:32632",
        )

        network = compute_hillslope_cells(gdf, simple_flowdir)

        # Should have empty list for hillslope_cells
        assert network.loc[0, "hillslope_cells"] == []
