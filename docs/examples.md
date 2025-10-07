# MOBIDICpy - Examples

## River Network Processing

### Basic Usage

Process a river network shapefile with default settings:

```python
from mobidic import process_river_network, export_network

# Process the network
network = process_river_network(
    shapefile_path="data/river_network.shp",
    id_field="REACH_ID"
)

# Export the result
export_network(network, "output/network.parquet")
```
