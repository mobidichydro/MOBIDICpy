# River Network Processing

The river network processing module provides a complete pipeline for building and analyzing river network topology, including Strahler ordering, reach joining, and routing parameter calculation.

## Overview

River network processing is a critical preprocessing step that:

- Builds network topology from shapefile geometry
- Computes Strahler stream ordering
- Optionally joins single-tributary reaches to simplify the network
- Calculates hydraulic and routing parameters (width, lag time, storage coefficient)
- Determines optimal calculation order for routing algorithms
- Exports processed networks for use in simulations

The implementation translates several MATLAB functions from the original MOBIDIC model:
`readshaperet.m`, `botte.m`, `bintree.m`, `crearete.m`, `strahler.m`, `preproc_ret.m`.

## Functions

### Network Processing

::: mobidic.preprocessing.river_network.process_river_network

### Network I/O

::: mobidic.preprocessing.river_network.export_network

::: mobidic.preprocessing.river_network.load_network

## Examples

### Complete Network Processing

```python
from mobidic import process_river_network, export_network

# Process river network from shapefile
network = process_river_network(
    shapefile_path="data/river_network.shp",
    reach_id_field="ID",
    join_single_tributaries=True,
    wcel=2.5,  # Celerity: 2.5 m/s
    Br0=5.0,   # Base width: 5 m
    NBr=0.5    # Width exponent
)

# Inspect the processed network
print(f"Total reaches: {len(network)}")
print(f"Max Strahler order: {network['strahler_order'].max()}")
print(f"Total length: {network['length_m'].sum() / 1000:.1f} km")

# Export to GeoParquet (recommended)
export_network(network, "output/network.parquet", format="parquet")

# Or export to Shapefile (for compatibility)
export_network(network, "output/network.shp", format="shapefile")
```

### Loading a Processed Network

```python
from mobidic import load_network

# Load previously processed network
network = load_network("output/network.parquet")

# Access network properties
terminal_reaches = network[network['downstream'].isna()]
print(f"Number of outlets: {len(terminal_reaches)}")

headwater_reaches = network[network['strahler_order'] == 1]
print(f"Number of headwater streams: {len(headwater_reaches)}")
```

### Analyzing Network Topology

```python
from mobidic import process_river_network
import pandas as pd

network = process_river_network("data/river_network.shp", reach_id_field="ID")

# Get summary statistics by Strahler order
summary = network.groupby('strahler_order').agg({
    'length_m': ['count', 'sum', 'mean'],
    'width_m': 'mean',
    'lag_time_s': 'mean'
})

print(summary)

# Find reaches with multiple upstream tributaries
junctions = network[network['upstream_2'].notna()]
print(f"Number of junctions: {len(junctions)}")

# Trace upstream from a specific reach
def get_upstream_network(network, reach_id):
    """Recursively find all upstream reaches."""
    upstream = []
    reach = network[network['reach_id'] == reach_id].iloc[0]

    for us_field in ['upstream_1', 'upstream_2']:
        us_id = reach[us_field]
        if pd.notna(us_id):
            upstream.append(us_id)
            upstream.extend(get_upstream_network(network, us_id))

    return upstream

# Get all reaches upstream of reach 100
upstream_reaches = get_upstream_network(network, 100)
print(f"Reaches upstream of 100: {len(upstream_reaches)}")
```

## Network Schema

Processed networks contain the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `reach_id` | int | Unique reach identifier |
| `geometry` | LineString | Reach geometry |
| `upstream_1` | int/NaN | First upstream reach ID |
| `upstream_2` | int/NaN | Second upstream reach ID (if junction) |
| `downstream` | int/NaN | Downstream reach ID (NaN for outlets) |
| `strahler_order` | int | Strahler stream order (1 = headwater) |
| `calc_order` | int | Calculation order for routing (lower first) |
| `length_m` | float | Reach length in meters |
| `width_m` | float | Channel width in meters |
| `lag_time_s` | float | Lag time in seconds |
| `storage_coeff` | float | Storage coefficient (dimensionless) |
| `n_manning` | float | Manning roughness coefficient |

## Strahler Ordering

The Strahler stream order is a hierarchical classification system:

1. **First-order streams** (order 1): Headwater streams with no upstream tributaries
2. **Higher-order streams**: When two streams of order *n* join, the result is order *n+1*
3. **Unequal joining**: When streams of different orders join, the downstream reach inherits the higher order

Example network:
```
    1 ─┐
       ├─ 2 ─┐
    1 ─┘     │
             ├─ 3 ─┐
          2 ─┘     │
                   ├─ 4 (outlet)
                3 ─┘
```

## Routing Parameters

Calculated parameters for channel routing:

- **Channel width**: `B = Br0 × order^NBr`
  - Uses power-law relationship with Strahler order
  - `Br0`: Base width for first-order streams
  - `NBr`: Width scaling exponent

- **Lag time**: `τ = L / wcel`
  - Time for a flow disturbance to travel the reach length
  - `L`: Reach length (m)
  - `wcel`: Wave celerity (m/s)

- **Storage coefficient**: `K = 0.5 × exp(-L/10000)`
  - Dimensionless attenuation factor
  - Decreases exponentially with reach length

- **Manning coefficient**: `n_manning`
  - Roughness parameter for friction calculations
  - Can be specified or estimated from channel properties

## Network Topology

### Supported Structures

- **Binary trees**: Maximum 2 upstream tributaries per reach
- **Multiple outlets**: Supports networks with more than one terminal reach
- **Disconnected networks**: Warns if multiple unconnected sub-networks exist

### Topology Validation

The processing pipeline validates:

- Reach connectivity (no dangling reaches)
- Non-binary junctions (warns if >2 upstream tributaries)
- Circular references (detects loops)
- Duplicate reach IDs

### Reach Joining

When `join_single_tributaries=True`:

1. Identifies reaches with exactly one upstream and one downstream reach
2. Merges these into the downstream reach
3. Recalculates length, routing parameters, and geometry
4. Simplifies the network while preserving topology

This is useful for:
- Reducing computational load
- Focusing on hydrologically significant junctions
- Matching observational data at specific locations

## Performance

Network processing performance (tested on Arno river basin):

- **Input**: 1,281 reaches, ~3,572 km total length
- **After joining**: 1,235 reaches (46 merged)
- **Processing time**: <2 seconds
- **Strahler orders**: 1-5
- **Memory usage**: Minimal (<50 MB)

Suitable for large networks (>10,000 reaches) with efficient geopandas operations.

## File Formats

### GeoParquet (Recommended)

- **Pros**: Fast I/O, efficient compression, preserves all data types, widely supported
- **Cons**: Requires `pyarrow` or `fastparquet`
- **Use**: Default format for processed networks

### Shapefile (Legacy)

- **Pros**: Universal compatibility, GIS software support
- **Cons**: Slower I/O, field name limitations (10 chars), limited data types
- **Use**: When compatibility with legacy tools is required

## Notes

- All lengths are in meters, times in seconds
- Coordinate reference system (CRS) is preserved from input shapefile
- Reach IDs must be unique integers
- Geometry must be LineString (not MultiLineString)
- Network must be topologically connected (warns if not)
