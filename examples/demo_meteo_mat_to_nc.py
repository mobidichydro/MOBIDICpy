"""Example script demonstrating meteorological data preprocessing.

This script shows how to convert MATLAB .mat meteorological data file to
CF-compliant NetCDF format for use in MOBIDIC simulations.
"""

from pathlib import Path
from mobidic import MeteoData, convert_mat_to_netcdf

# Example 1: Direct conversion using convenience function
print("=" * 80)
print("Example 1: Direct MAT to NetCDF conversion")
print("=" * 80)

mat_file = Path("examples/Arno/meteodata/meteodata.mat")
output_file = Path("examples/Arno/meteodata/meteodata.nc")

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

print("\n" + "=" * 80)
print("Meteo preprocessing complete!")
print("=" * 80)
