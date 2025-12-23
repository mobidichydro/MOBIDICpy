"""Tests for reservoir preprocessing module."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon, LineString

from mobidic.preprocessing.reservoirs import (
    Reservoir,
    Reservoirs,
    _rasterize_reservoir_polygon,
    _find_reservoir_reaches,
    _interpolate_volume_at_stage,
    process_reservoirs,
)


@pytest.fixture
def simple_reservoir():
    """Create a simple reservoir object for testing."""
    return Reservoir(
        id=1,
        z_max=250.0,
        name="Test Reservoir",
        basin_pixels=np.array([10, 11, 12, 20, 21, 22]),
        inlet_reaches=np.array([1, 2]),
        outlet_reach=5,
        stage_storage_curve=pd.DataFrame({"stage_m": [240, 245, 250], "volume_m3": [1000, 5000, 10000]}),
        period_times={"000": "2020-01-01T00:00:00", "001": "2020-06-01T00:00:00"},
        stage_discharge_h={"000": [240.0, 245.0, 250.0], "001": [240.0, 245.0, 250.0]},
        stage_discharge_q={"000": [0.0, 5.0, 10.0], "001": [0.0, 10.0, 20.0]},
        initial_volume=5000.0,
        date_start=pd.Timestamp("2020-01-01"),
        geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    )


@pytest.fixture
def simple_reservoir_polygon():
    """Create a simple reservoir polygon."""
    return Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])


@pytest.fixture
def simple_river_network():
    """Create a simple river network for testing reservoir-reach mapping.

    Network structure:
        R1 (upstream) -> R2 (middle) -> R3 (outlet)
    """
    reaches = {
        "mobidic_id": [0, 1, 2],
        "REACH_ID": [1, 2, 3],
        "upstream_1": [np.nan, 0, 1],
        "upstream_2": [np.nan, np.nan, np.nan],
        "downstream": [1, 2, -1],
        "calc_order": [1, 2, 3],
        "geometry": [
            LineString([(50, 150), (100, 150)]),  # R1: upstream, intersects reservoir
            LineString([(100, 150), (150, 150)]),  # R2: middle, intersects reservoir
            LineString([(150, 150), (250, 150)]),  # R3: outlet, intersects reservoir
        ],
    }

    return gpd.GeoDataFrame(reaches, crs="EPSG:3003")


@pytest.fixture
def reservoir_shapefile_gdf():
    """Create a GeoDataFrame mimicking reservoir shapefile structure."""
    data = {
        "id": [1, 2],
        "name": ["Reservoir A", "Reservoir B"],
        "zmax": [250.0, 260.0],
        "date_start": ["2020-01-01", "2020-01-01"],
        "geometry": [
            Polygon([(100, 100), (200, 100), (200, 200), (100, 200)]),
            Polygon([(300, 300), (400, 300), (400, 400), (300, 400)]),
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:3003")


@pytest.fixture
def stage_storage_csv_data():
    """Create sample stage-storage curve data."""
    return pd.DataFrame(
        {
            "reservoir_id": [1, 1, 1, 1, 2, 2, 2],
            "stage_m": [240.0, 245.0, 250.0, 255.0, 250.0, 255.0, 260.0],
            "volume_m3": [1000, 5000, 10000, 20000, 2000, 8000, 15000],
        }
    )


@pytest.fixture
def regulation_curves_csv_data():
    """Create sample regulation curves data."""
    return pd.DataFrame(
        {
            "reservoir_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "regulation_name": [
                "winter",
                "winter",
                "winter",
                "summer",
                "summer",
                "summer",
                "winter",
                "winter",
                "winter",
            ],
            "stage_m": [240.0, 245.0, 250.0, 240.0, 245.0, 250.0, 250.0, 255.0, 260.0],
            "discharge_m3s": [0.0, 5.0, 10.0, 0.0, 10.0, 20.0, 0.0, 10.0, 20.0],
        }
    )


@pytest.fixture
def regulation_schedule_csv_data():
    """Create sample regulation schedule data."""
    return pd.DataFrame(
        {
            "reservoir_id": [1, 1, 2, 2],
            "start_date": ["2020-01-01", "2020-06-01", "2020-01-01", "2020-06-01"],
            "end_date": ["2020-05-31", "2020-12-31", "2020-05-31", "2020-12-31"],
            "regulation_name": ["winter", "summer", "winter", "winter"],
        }
    )


@pytest.fixture
def initial_volumes_csv_data():
    """Create sample initial volumes data."""
    return pd.DataFrame({"reservoir_id": [1, 2], "volume_m3": [5000.0, 7000.0]})


# Tests for Reservoir dataclass


def test_reservoir_creation(simple_reservoir):
    """Test basic Reservoir object creation."""
    assert simple_reservoir.id == 1
    assert simple_reservoir.z_max == 250.0
    assert simple_reservoir.name == "Test Reservoir"
    assert len(simple_reservoir.basin_pixels) == 6
    assert len(simple_reservoir.inlet_reaches) == 2
    assert simple_reservoir.outlet_reach == 5


def test_reservoir_minimal_creation():
    """Test Reservoir creation with minimal required fields."""
    res = Reservoir(id=1, z_max=250.0)
    assert res.id == 1
    assert res.z_max == 250.0
    assert res.name == ""
    assert res.basin_pixels is None
    assert res.inlet_reaches is None


# Tests for Reservoirs container class


def test_reservoirs_init_empty():
    """Test Reservoirs initialization with empty list."""
    reservoirs = Reservoirs([])
    assert len(reservoirs) == 0
    assert reservoirs.metadata == {}


def test_reservoirs_init_with_data(simple_reservoir):
    """Test Reservoirs initialization with data."""
    reservoirs = Reservoirs([simple_reservoir], metadata={"crs": "EPSG:3003"})
    assert len(reservoirs) == 1
    assert reservoirs.metadata["crs"] == "EPSG:3003"


def test_reservoirs_len(simple_reservoir):
    """Test __len__ method."""
    reservoirs = Reservoirs([simple_reservoir, simple_reservoir])
    assert len(reservoirs) == 2


def test_reservoirs_getitem(simple_reservoir):
    """Test __getitem__ method."""
    reservoirs = Reservoirs([simple_reservoir])
    res = reservoirs[0]
    assert res.id == 1
    assert res.name == "Test Reservoir"


def test_reservoirs_to_geodataframe_empty():
    """Test converting empty Reservoirs to GeoDataFrame."""
    reservoirs = Reservoirs([])
    gdf = reservoirs.to_geodataframe()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 0
    # Check that expected columns exist
    expected_cols = [
        "id",
        "name",
        "z_max",
        "basin_pixels",
        "inlet_reaches",
        "outlet_reach",
        "period_times",
        "stage_discharge_h",
        "stage_discharge_q",
        "initial_volume",
        "stage_storage_curve",
        "geometry",
    ]
    for col in expected_cols:
        assert col in gdf.columns


def test_reservoirs_to_geodataframe_with_data(simple_reservoir):
    """Test converting Reservoirs to GeoDataFrame with data."""
    reservoirs = Reservoirs([simple_reservoir], metadata={"crs": "EPSG:3003"})
    gdf = reservoirs.to_geodataframe()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.crs == "EPSG:3003"

    # Check data conversion
    row = gdf.iloc[0]
    assert row["id"] == 1
    assert row["name"] == "Test Reservoir"
    assert row["z_max"] == 250.0
    assert row["basin_pixels"] == [10, 11, 12, 20, 21, 22]
    assert row["inlet_reaches"] == [1, 2]
    assert row["outlet_reach"] == 5
    assert isinstance(row["stage_storage_curve"], list)
    assert isinstance(row["period_times"], dict)


def test_reservoirs_save_parquet(simple_reservoir, tmp_path):
    """Test saving Reservoirs to Parquet format."""
    reservoirs = Reservoirs([simple_reservoir], metadata={"crs": "EPSG:3003"})
    output_path = tmp_path / "reservoirs.parquet"

    reservoirs.save(output_path, format="parquet")

    assert output_path.exists()


def test_reservoirs_save_invalid_format(simple_reservoir, tmp_path):
    """Test that invalid save format raises ValueError."""
    reservoirs = Reservoirs([simple_reservoir])
    output_path = tmp_path / "reservoirs.csv"

    with pytest.raises(ValueError, match="Unsupported format"):
        reservoirs.save(output_path, format="csv")


def test_reservoirs_load_parquet(simple_reservoir, tmp_path):
    """Test loading Reservoirs from Parquet file."""
    # Save first
    reservoirs = Reservoirs([simple_reservoir], metadata={"crs": "EPSG:3003"})
    output_path = tmp_path / "reservoirs.parquet"
    reservoirs.save(output_path)

    # Load
    loaded = Reservoirs.load(output_path)

    assert len(loaded) == 1
    assert loaded.metadata["crs"] == "EPSG:3003"

    res = loaded[0]
    assert res.id == 1
    assert res.z_max == 250.0
    assert res.name == "Test Reservoir"
    assert len(res.basin_pixels) == 6
    assert len(res.inlet_reaches) == 2
    assert res.outlet_reach == 5
    assert isinstance(res.stage_storage_curve, pd.DataFrame)


def test_reservoirs_roundtrip_parquet(simple_reservoir, tmp_path):
    """Test save/load roundtrip preserves data."""
    original = Reservoirs([simple_reservoir], metadata={"crs": "EPSG:3003"})
    path = tmp_path / "reservoirs.parquet"

    # Save and load
    original.save(path)
    loaded = Reservoirs.load(path)

    # Compare
    assert len(loaded) == len(original)
    orig_res = original[0]
    load_res = loaded[0]

    assert load_res.id == orig_res.id
    assert load_res.z_max == orig_res.z_max
    assert load_res.name == orig_res.name
    assert np.array_equal(load_res.basin_pixels, orig_res.basin_pixels)
    assert np.array_equal(load_res.inlet_reaches, orig_res.inlet_reaches)
    assert load_res.outlet_reach == orig_res.outlet_reach
    assert load_res.initial_volume == orig_res.initial_volume


# Tests for _rasterize_reservoir_polygon


def test_rasterize_reservoir_polygon_simple():
    """Test rasterizing a simple square polygon."""
    # Create a small polygon in a 10x10 grid
    # Grid covers area (0, 0) to (100, 100) with cellsize=10
    polygon = Polygon([(20, 20), (40, 20), (40, 40), (20, 40)])
    grid_shape = (10, 10)
    xllcorner = 0.0
    yllcorner = 0.0
    cellsize = 10.0

    basin_pixels = _rasterize_reservoir_polygon(polygon, grid_shape, xllcorner, yllcorner, cellsize)

    # Should have some pixels
    assert len(basin_pixels) > 0
    # All indices should be valid
    assert np.all(basin_pixels >= 0)
    assert np.all(basin_pixels < 100)  # 10*10 grid


def test_rasterize_reservoir_polygon_all_touched():
    """Test that all_touched=True includes partially intersected cells."""
    # Small polygon that should touch 4 cells
    polygon = Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])
    grid_shape = (10, 10)
    xllcorner = 0.0
    yllcorner = 0.0
    cellsize = 10.0

    basin_pixels = _rasterize_reservoir_polygon(polygon, grid_shape, xllcorner, yllcorner, cellsize)

    # With all_touched=True, should get at least 4 cells
    assert len(basin_pixels) >= 4


def test_rasterize_reservoir_polygon_outside_grid():
    """Test rasterizing polygon completely outside grid."""
    # Polygon outside grid bounds
    polygon = Polygon([(200, 200), (300, 200), (300, 300), (200, 300)])
    grid_shape = (10, 10)
    xllcorner = 0.0
    yllcorner = 0.0
    cellsize = 10.0

    basin_pixels = _rasterize_reservoir_polygon(polygon, grid_shape, xllcorner, yllcorner, cellsize)

    # Should have no pixels
    assert len(basin_pixels) == 0


# Tests for _find_reservoir_reaches


def test_find_reservoir_reaches_simple(simple_reservoir_polygon, simple_river_network):
    """Test finding inlet and outlet reaches for a reservoir."""
    reservoir = Reservoir(id=1, z_max=250.0, geometry=simple_reservoir_polygon)

    inlet_reaches, outlet_reach = _find_reservoir_reaches(reservoir, simple_river_network)

    # Should find an outlet reach
    assert outlet_reach >= 0
    # Outlet should be one of the reaches that intersects the polygon
    assert outlet_reach in [0, 1, 2]


def test_find_reservoir_reaches_with_upstream():
    """Test that inlet reaches are upstream of outlet."""
    # Create reservoir that intersects R2 and R3
    reservoir_polygon = Polygon([(125, 125), (175, 125), (175, 175), (125, 175)])
    reservoir = Reservoir(id=1, z_max=250.0, geometry=reservoir_polygon)

    # Create network where R2 and R3 intersect reservoir
    reaches = {
        "mobidic_id": [0, 1, 2],
        "REACH_ID": [1, 2, 3],
        "upstream_1": [np.nan, 0, 1],
        "upstream_2": [np.nan, np.nan, np.nan],
        "downstream": [1, 2, -1],
        "calc_order": [1, 2, 3],
        "geometry": [
            LineString([(100, 150), (125, 150)]),  # R1: ends before reservoir
            LineString([(125, 150), (175, 150)]),  # R2: through reservoir
            LineString([(175, 150), (250, 150)]),  # R3: through reservoir (highest calc_order)
        ],
    }
    network = gpd.GeoDataFrame(reaches, crs="EPSG:3003")

    inlet_reaches, outlet_reach = _find_reservoir_reaches(reservoir, network)

    # Outlet should be the reach with highest calc_order among intersecting reaches
    # Both R2 (mobidic_id=1) and R3 (mobidic_id=2) intersect, so R3 with calc_order=3 is outlet
    assert outlet_reach == 2
    # Inlet should be upstream reach of outlet (R2, mobidic_id=1)
    assert 1 in inlet_reaches


def test_find_reservoir_reaches_no_intersection():
    """Test finding reaches when reservoir doesn't intersect network."""
    # Reservoir far from network
    reservoir_polygon = Polygon([(500, 500), (600, 500), (600, 600), (500, 600)])
    reservoir = Reservoir(id=1, z_max=250.0, geometry=reservoir_polygon)

    reaches = {
        "mobidic_id": [0, 1],
        "upstream_1": [np.nan, 0],
        "upstream_2": [np.nan, np.nan],
        "downstream": [1, -1],
        "calc_order": [1, 2],
        "geometry": [LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
    }
    network = gpd.GeoDataFrame(reaches, crs="EPSG:3003")

    inlet_reaches, outlet_reach = _find_reservoir_reaches(reservoir, network)

    # Should find no reaches
    assert outlet_reach == -1
    assert len(inlet_reaches) == 0


def test_find_reservoir_reaches_multiple_upstreams():
    """Test finding multiple inlet reaches."""
    # Reservoir that intersects junction
    reservoir_polygon = Polygon([(175, 100), (225, 100), (225, 250), (175, 250)])
    reservoir = Reservoir(id=1, z_max=250.0, geometry=reservoir_polygon)

    # Create Y-shaped network
    reaches = {
        "mobidic_id": [0, 1, 2],
        "upstream_1": [np.nan, np.nan, 0],
        "upstream_2": [np.nan, np.nan, 1],
        "downstream": [2, 2, -1],
        "calc_order": [1, 1, 3],
        "geometry": [
            LineString([(100, 220), (200, 220)]),  # R1: left tributary
            LineString([(100, 180), (200, 180)]),  # R2: right tributary
            LineString([(200, 180), (200, 220), (300, 200)]),  # R3: junction through reservoir
        ],
    }
    network = gpd.GeoDataFrame(reaches, crs="EPSG:3003")

    inlet_reaches, outlet_reach = _find_reservoir_reaches(reservoir, network)

    # Outlet should be R3 (highest calc_order)
    assert outlet_reach == 2
    # Should have 2 inlet reaches
    assert len(inlet_reaches) == 2
    assert 0 in inlet_reaches
    assert 1 in inlet_reaches


# Tests for process_reservoirs integration


def test_process_reservoirs_integration(
    reservoir_shapefile_gdf,
    stage_storage_csv_data,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    initial_volumes_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test full reservoir processing pipeline."""
    # Save test data to CSV files
    shp_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shp_path)

    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    initial_volumes_path = tmp_path / "initial_volumes.csv"
    initial_volumes_csv_data.to_csv(initial_volumes_path, index=False)

    # Process reservoirs
    reservoirs = process_reservoirs(
        shapefile_path=shp_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=50.0,
    )

    # Should process 2 reservoirs
    assert len(reservoirs) == 2

    # Check first reservoir
    res1 = reservoirs[0]
    assert res1.id == 1
    assert res1.name == "Reservoir A"
    assert res1.z_max == 250.0
    assert res1.initial_volume == 5000.0
    assert res1.basin_pixels is not None
    assert len(res1.basin_pixels) > 0
    assert res1.stage_storage_curve is not None
    assert len(res1.stage_storage_curve) == 4  # 4 rows for reservoir 1
    assert res1.period_times is not None
    assert res1.stage_discharge_h is not None
    assert res1.stage_discharge_q is not None


def test_process_reservoirs_empty_shapefile(
    stage_storage_csv_data,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    initial_volumes_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test processing with empty shapefile."""
    # Create empty shapefile
    empty_gdf = gpd.GeoDataFrame(columns=["id", "name", "zmax", "geometry"], crs="EPSG:3003")
    shp_path = tmp_path / "empty_reservoirs.shp"
    empty_gdf.to_file(shp_path)

    # Create CSV files
    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    initial_volumes_path = tmp_path / "initial_volumes.csv"
    initial_volumes_csv_data.to_csv(initial_volumes_path, index=False)

    # Process
    reservoirs = process_reservoirs(
        shapefile_path=shp_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=50.0,
    )

    # Should return empty container
    assert len(reservoirs) == 0


def test_process_reservoirs_missing_initial_volume(
    reservoir_shapefile_gdf,
    stage_storage_csv_data,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test processing when initial volume is missing (should auto-calculate from curve)."""
    # Save files
    shp_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shp_path)

    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    # Create initial volumes with only reservoir 2
    initial_volumes = pd.DataFrame({"reservoir_id": [2], "volume_m3": [7000.0]})
    initial_volumes_path = tmp_path / "initial_volumes.csv"
    initial_volumes.to_csv(initial_volumes_path, index=False)

    # Process
    reservoirs = process_reservoirs(
        shapefile_path=shp_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=50.0,
    )

    # Reservoir 1 should have initial_volume auto-calculated from z_max=250 → 10000 m³
    res1 = reservoirs[0]
    assert res1.id == 1
    assert res1.initial_volume == pytest.approx(10000.0)


def test_process_reservoirs_missing_stage_storage(
    reservoir_shapefile_gdf,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    initial_volumes_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test that reservoir without stage-storage curve is skipped."""
    # Save files
    shp_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shp_path)

    # Stage-storage only for reservoir 2
    stage_storage = pd.DataFrame(
        {"reservoir_id": [2, 2, 2], "stage_m": [250.0, 255.0, 260.0], "volume_m3": [2000, 8000, 15000]}
    )
    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    initial_volumes_path = tmp_path / "initial_volumes.csv"
    initial_volumes_csv_data.to_csv(initial_volumes_path, index=False)

    # Process
    reservoirs = process_reservoirs(
        shapefile_path=shp_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=50.0,
    )

    # Should only have reservoir 2
    assert len(reservoirs) == 1
    assert reservoirs[0].id == 2


def test_process_reservoirs_missing_regulation_schedule(
    reservoir_shapefile_gdf,
    stage_storage_csv_data,
    regulation_curves_csv_data,
    initial_volumes_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test that reservoir without regulation schedule is skipped."""
    # Save files
    shp_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shp_path)

    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    # Schedule only for reservoir 2
    regulation_schedule = pd.DataFrame(
        {
            "reservoir_id": [2, 2],
            "start_date": ["2020-01-01", "2020-06-01"],
            "end_date": ["2020-05-31", "2020-12-31"],
            "regulation_name": ["winter", "winter"],
        }
    )
    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule.to_csv(regulation_schedule_path, index=False)

    initial_volumes_path = tmp_path / "initial_volumes.csv"
    initial_volumes_csv_data.to_csv(initial_volumes_path, index=False)

    # Process
    reservoirs = process_reservoirs(
        shapefile_path=shp_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=50.0,
    )

    # Should only have reservoir 2
    assert len(reservoirs) == 1
    assert reservoirs[0].id == 2


def test_process_reservoirs_regulation_period_arrays(
    reservoir_shapefile_gdf,
    stage_storage_csv_data,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    initial_volumes_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test that regulation period arrays are correctly structured."""
    # Save files
    shp_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shp_path)

    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    initial_volumes_path = tmp_path / "initial_volumes.csv"
    initial_volumes_csv_data.to_csv(initial_volumes_path, index=False)

    # Process
    reservoirs = process_reservoirs(
        shapefile_path=shp_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=50.0,
    )

    res1 = reservoirs[0]

    # Check period_times structure (dict with zero-padded string keys)
    assert isinstance(res1.period_times, dict)
    assert "000" in res1.period_times
    assert "001" in res1.period_times
    assert len(res1.period_times) == 2

    # Check stage_discharge arrays (dict with zero-padded string keys)
    assert isinstance(res1.stage_discharge_h, dict)
    assert isinstance(res1.stage_discharge_q, dict)
    assert "000" in res1.stage_discharge_h
    assert "001" in res1.stage_discharge_h

    # Each period should have same-length arrays
    assert len(res1.stage_discharge_h["000"]) == len(res1.stage_discharge_q["000"])
    assert len(res1.stage_discharge_h["001"]) == len(res1.stage_discharge_q["001"])


def test_interpolate_volume_clamping(stage_storage_csv_data):
    """Test volume interpolation with z_max beyond curve bounds."""
    # Get reservoir 1 curve: stages [240, 245, 250, 255], volumes [1000, 5000, 10000, 20000]
    ss_curve = stage_storage_csv_data[stage_storage_csv_data["reservoir_id"] == 1].copy()
    ss_curve = ss_curve.drop(columns=["reservoir_id"])

    # Test exact match in range
    vol_250 = _interpolate_volume_at_stage(ss_curve, 250.0)
    assert vol_250 == 10000.0

    # Test interpolation in range
    vol_247_5 = _interpolate_volume_at_stage(ss_curve, 247.5)
    assert vol_247_5 == pytest.approx(7500.0)  # Linear interpolation between 245 (5000) and 250 (10000)

    # Test above max (should clamp to max volume)
    vol_260 = _interpolate_volume_at_stage(ss_curve, 260.0)
    assert vol_260 == 20000.0  # Clamped to max

    # Test below min (should clamp to min volume)
    vol_230 = _interpolate_volume_at_stage(ss_curve, 230.0)
    assert vol_230 == 1000.0  # Clamped to min


def test_process_reservoirs_auto_calculate_volumes(
    reservoir_shapefile_gdf,
    stage_storage_csv_data,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test that initial volumes are auto-calculated when CSV path is None."""
    # Setup files (without initial_volumes.csv)
    shapefile_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shapefile_path)

    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    # Process reservoirs with initial_volumes_path=None
    reservoirs = process_reservoirs(
        shapefile_path=shapefile_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=None,  # <-- Test None path
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=10.0,
    )

    # Should process 2 reservoirs
    assert len(reservoirs) == 2

    # Reservoir 1: z_max=250, should interpolate to 10000 m³
    res1 = reservoirs[0]
    assert res1.id == 1
    assert res1.initial_volume == pytest.approx(10000.0)

    # Reservoir 2: z_max=260, should interpolate to 15000 m³
    res2 = reservoirs[1]
    assert res2.id == 2
    assert res2.initial_volume == pytest.approx(15000.0)


def test_process_reservoirs_partial_volumes_csv(
    reservoir_shapefile_gdf,
    stage_storage_csv_data,
    regulation_curves_csv_data,
    regulation_schedule_csv_data,
    simple_river_network,
    tmp_path,
):
    """Test that missing reservoirs in CSV are auto-calculated."""
    # Setup files
    shapefile_path = tmp_path / "reservoirs.shp"
    reservoir_shapefile_gdf.to_file(shapefile_path)

    stage_storage_path = tmp_path / "stage_storage.csv"
    stage_storage_csv_data.to_csv(stage_storage_path, index=False)

    regulation_curves_path = tmp_path / "regulation_curves.csv"
    regulation_curves_csv_data.to_csv(regulation_curves_path, index=False)

    regulation_schedule_path = tmp_path / "regulation_schedule.csv"
    regulation_schedule_csv_data.to_csv(regulation_schedule_path, index=False)

    # Create partial initial volumes CSV (only reservoir 1, missing reservoir 2)
    partial_volumes = pd.DataFrame({"reservoir_id": [1], "volume_m3": [3333.0]})
    initial_volumes_path = tmp_path / "initial_volumes.csv"
    partial_volumes.to_csv(initial_volumes_path, index=False)

    # Process reservoirs
    reservoirs = process_reservoirs(
        shapefile_path=shapefile_path,
        stage_storage_path=stage_storage_path,
        regulation_curves_path=regulation_curves_path,
        regulation_schedule_path=regulation_schedule_path,
        initial_volumes_path=initial_volumes_path,  # <-- Partial CSV
        network=simple_river_network,
        grid_shape=(10, 10),
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=10.0,
    )

    # Should process 2 reservoirs
    assert len(reservoirs) == 2

    # Reservoir 1: should use CSV value
    res1 = reservoirs[0]
    assert res1.id == 1
    assert res1.initial_volume == pytest.approx(3333.0)

    # Reservoir 2: should auto-calculate (z_max=260 → 15000 m³)
    res2 = reservoirs[1]
    assert res2.id == 2
    assert res2.initial_volume == pytest.approx(15000.0)
