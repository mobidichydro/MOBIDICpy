# Report I/O

The report I/O module provides functions to save and load discharge and lateral inflow time series in Parquet format for efficient storage and analysis.

## Overview

Report files store:

- **Discharge time series**: River discharge for selected reaches over simulation period
- **Lateral inflow time series**: Hillslope contributions to river reaches
- **Time index**: Datetime index for all time steps
- **Reach selection**: Configurable subset (all, outlets, custom list)
- **Metadata**: Simulation details saved as separate JSON file

Parquet format offers:

- **High compression**: ~10-50× smaller than CSV
- **Fast I/O**: Columnar storage optimized for analytics
- **Type preservation**: Maintains datetime and numeric types
- **Integration**: Compatible with pandas, Dask, Apache Spark

## Functions

::: mobidic.io.report.save_discharge_report

::: mobidic.io.report.load_discharge_report

::: mobidic.io.report.save_lateral_inflow_report


## Design Features

- **Efficient storage**: Parquet columnar format with compression
- **Fast loading**: Optimized for time series analysis
- **Flexible selection**: All reaches, outlets only, or custom list
- **Type preservation**: Maintains datetime and float64 types
- **Metadata support**: Optional JSON metadata file
- **pandas integration**: Seamless integration with pandas workflows

## Integration with other tools

### Load in R

```r
library(arrow)
df <- read_parquet("discharge.parquet")
head(df)
```

### Load in Apache Spark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df_spark = spark.read.parquet("discharge.parquet")
df_spark.show()
```

### Load in Dask

```python
import dask.dataframe as dd

df_dask = dd.read_parquet("discharge.parquet")
print(df_dask.head())
```

## References

**File format**:

- Apache Parquet with Snappy compression
- Uses PyArrow engine for reading/writing
- Compatible with pandas, Dask, Spark, R (arrow package)

**Related modules**:

- [Simulation](simulation.md) - Generates time series for reports
- [State I/O](state.md) - Spatial state snapshots (NetCDF format)
