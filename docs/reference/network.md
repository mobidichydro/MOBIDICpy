# River network processing

The river network processing module provides a toolkit for building and analyzing river network topology, including Strahler ordering, reach joining, and routing parameter calculation.

## Overview

River network processing is a required preprocessing step that:

- Builds network topology from a shapefile geometry
- Enforces binary tree structure (maximum 2 upstream tributaries per reach)
- Computes Strahler stream ordering
- Optionally joins single-tributary reaches to simplify the network
- Calculates hydraulic and routing parameters (width, lag time, storage coefficient)
- Determines optimal calculation order for routing algorithms
- Exports processed networks for use in simulations

## Functions

### River network processing

::: mobidic.preprocessing.river_network.process_river_network

### River network I/O

::: mobidic.preprocessing.io.save_network

::: mobidic.preprocessing.io.load_network

## Examples

### Complete river network processing

```python
from mobidic import process_river_network, save_network

# Process river network from shapefile
network = process_river_network(
    shapefile_path="data/river_network.shp",
    join_single_tributaries=True,
    routing_params={
        "wcel": 2.5,  # Celerity: 2.5 m/s
        "Br0": 5.0,   # Base width: 5 m
        "NBr": 0.5,   # Width exponent
        "n_Man": 0.03  # Manning coefficient
    }
)

# Inspect the processed network
print(f"Total reaches: {len(network)}")
print(f"Max Strahler order: {network['strahler_order'].max()}")
print(f"Total length: {network['length_m'].sum() / 1000:.1f} km")

# Export to GeoParquet (recommended)
save_network(network, "output/network.parquet", format="parquet")

# Or export to Shapefile (for backward compatibility)
save_network(network, "output/network.shp", format="shapefile")
```

### Loading a processed Network

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

### Analyzing river network topology

```python
from mobidic import process_river_network
import pandas as pd

network = process_river_network("data/river_network.shp")

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
def get_upstream_network(network, mobidic_id):
    """Recursively find all upstream reaches."""
    upstream = []
    reach = network[network['mobidic_id'] == mobidic_id].iloc[0]

    for us_field in ['upstream_1', 'upstream_2']:
        us_id = reach[us_field]
        if pd.notna(us_id):
            upstream.append(us_id)
            upstream.extend(get_upstream_network(network, us_id))

    return upstream

# Get all reaches upstream of reach 100
upstream_reaches = get_upstream_network(network, 100)
print(f"Reaches upstream of reach 100: {len(upstream_reaches)}")
```

## River network schema

Processed networks contain the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `mobidic_id` | int | Internal reach identifier (0-indexed for topology) |
| `geometry` | LineString | Reach geometry |
| `upstream_1` | int/NaN | First upstream reach ID (references mobidic_id) |
| `upstream_2` | int/NaN | Second upstream reach ID if junction (references mobidic_id) |
| `downstream` | int/NaN | Downstream reach ID (references mobidic_id, NaN for outlets) |
| `strahler_order` | int | Strahler stream order (1 = headwater) |
| `calc_order` | int | Calculation order for routing (lower first) |
| `length_m` | float | Reach length in meters |
| `width_m` | float | Channel width in meters |
| `lag_time_s` | float | Lag time in seconds (used as K in routing) |
| `n_manning` | float | Manning roughness coefficient |

**Note**: All original shapefile fields are preserved in the output. The `mobidic_id` field is added for internal topology management and is separate from any original ID fields in the shapefile.

## Strahler ordering

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

## Routing parameters

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

## Processing pipeline

The river network processing follows these steps in order:

1. **Read shapefile**: Load river network geometry and attributes
2. **Build topology**: Identify upstream/downstream connections by matching reach endpoints
3. **Enforce binary tree**: Add fictitious short reaches for nodes with >2 tributaries
4. **Compute Strahler order**: Assign hierarchical stream orders
5. **Join single tributaries** (optional): Merge linear reach sequences
6. **Calculate routing parameters**: Compute width, lag time, storage coefficient
7. **Determine calculation order**: Set optimal processing sequence for routing

### Binary tree enforcement

Networks may have junctions where more than 2 reaches join. MOBIDIC requires binary trees (maximum 2 upstream tributaries per reach) for its routing algorithms. The `_enforce_binary_tree()` function:

- Identifies nodes with >2 upstream reaches
- Adds fictitious short reaches (0.1m length) to split non-binary junctions
- Updates topology to maintain network connectivity

**Example**: A 3-way junction (reaches A, B, C joining at node N flowing to reach D):

```
Before:              After:
  A ─┐                 A ─┐
  B ─┼─ N ─> D         B ─┼─ N1 ─> F ─> N ─> D
  C ─┘                 C ─┘         (fictitious)
```

This step typically adds a small number of fictitious reaches.

## River network topology

### Supported Structures

- **Binary trees**: Maximum 2 upstream tributaries per reach (enforced automatically)
- **Multiple outlets**: Supports networks with more than one terminal reach
- **Disconnected networks**: Warns if multiple unconnected sub-networks exist

### Topology Validation

The processing pipeline validates:

- Reach connectivity (no dangling reaches)
- Binary tree structure (automatically enforced)
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

**Note**: Joining is applied *after* binary tree enforcement, so some fictitious reaches may be merged.


## File Formats

### GeoParquet (Recommended)

- **Pros**: Fast I/O, efficient compression, preserves all data types, widely supported
- **Cons**: Requires `pyarrow` (default) or `fastparquet`
- **Use**: Default format for processed networks

### Shapefile (Legacy)

- **Pros**: Universal compatibility, GIS software support
- **Cons**: Slower I/O, field name limitations (10 chars), limited data types
- **Use**: When compatibility with legacy tools is required

## Notes

- All lengths are in meters, times in seconds
- Coordinate reference system (CRS) is preserved from input shapefile
- Original shapefile fields are preserved; `mobidic_id` is added for topology
- Geometry must be LineString (not MultiLineString)
- Network must be topologically connected (warns if not)
- Fictitious reaches (added during binary tree enforcement) contain only MOBIDIC-specific fields
