"""Example script demonstrating meteorological data preprocessing.

This script shows how to convert MATLAB .mat meteorological data file to
CF-compliant NetCDF format for use in MOBIDIC simulations.
"""

import sys
from pathlib import Path

from mobidic import MeteoData, convert_mat_to_netcdf
from mobidic.utils import configure_logger

# Configure logger using centralized configuration
configure_logger(level="DEBUG")

# Example 1: Direct conversion using convenience function
print("=" * 80)
print("Example 1: Direct MAT to NetCDF conversion")
print("=" * 80)

mat_file = Path(__file__).parent.parent / "datasets" / "Arno_event_Nov_2023" / "meteodata" / "meteodata.mat"
output_file = Path(__file__).parent / "meteodata.nc"

convert_mat_to_netcdf(
    mat_file,
    output_file,
    compression_level=4,
    add_metadata={
        "basin": "Arno",
        "basin_id": "Event_November_2023",
        "description": "Meteorological forcing data for Arno basin flood event",
    },
)

print("\n")

# Example 2: Using MeteoData class for more control
print("=" * 80)
print("Example 2: Loading and inspecting meteo data before saving")
print("=" * 80)

# Load from MAT file
meteo_data = MeteoData.from_mat(mat_file)

# Inspect loaded data
print(f"\n{meteo_data}")
print(f"\nDate range: {meteo_data.start_date} to {meteo_data.end_date}")
print("\nStation counts by variable:")
for var_name, stations in meteo_data.stations.items():
    print(f"  {var_name}: {len(stations)} stations")

# Examine first precipitation station
if len(meteo_data.stations["precipitation"]) > 0:
    first_station = meteo_data.stations["precipitation"][0]
    print("\nFirst precipitation station:")
    print(f"  Code: {first_station['code']}")
    print(f"  Location: ({first_station['x']:.2f}, {first_station['y']:.2f})")
    print(f"  Elevation: {first_station['elevation']:.1f} m")
    print(f"  Data points: {len(first_station['data'])}")

print("\n")

# Example 3: Reading from NetCDF just written
print("=" * 80)
print("Example 3: Loading data from NetCDF file")
print("=" * 80)

# Load from NetCDF file
meteo_from_nc = MeteoData.from_netcdf(output_file)

# Verify data
print(f"\n{meteo_from_nc}")
print(f"\nDate range: {meteo_from_nc.start_date} to {meteo_from_nc.end_date}")
print("\nStation counts by variable:")
for var_name, stations in meteo_from_nc.stations.items():
    print(f"  {var_name}: {len(stations)} stations")

# Compare with original
print("\nVerification:")
print(f"  Variables match: {set(meteo_data.variables) == set(meteo_from_nc.variables)}")
for var in meteo_data.variables:
    orig_count = len(meteo_data.stations[var])
    nc_count = len(meteo_from_nc.stations[var])
    print(f"  {var}: {orig_count} stations (MAT) vs {nc_count} stations (NetCDF)")

print("\n" + "=" * 80)
print("Meteo preprocessing complete!")
print("=" * 80)
