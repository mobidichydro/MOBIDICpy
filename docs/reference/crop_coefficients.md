# Crop coefficients (Kc / CLC)

This module supports the **FAO single crop coefficient** method to adjust the evapotranspiration (ET) for different land-cover types. Monthly $K_c$ values are mapped to **Corine Land Cover (CLC) level 3** codes, using default mapping values (see `data/kc_clc_mapping.csv` file). The user can provide a custom mapping file if desired. The $K_c$ factor is applied to the turbulent exchange coefficient $C_H$ in the energy balance calculation, ensuring that the resulting surface temperature and ET are consistent with the land-cover type.

## Background

The FAO-56 single crop coefficient methodology defines the crop evapotranspiration as:

$$ET_c = K_c \cdot ET_0$$

where:

- $ET_0$ is the reference evapotranspiration (calculated for a reference grass surface)
- $K_c$ is the crop coefficient that adjusts $ET_0$ to account for the specific vegetation type and growth stage.

$K_c$ depends on:

- **Vegetation structure**: leaf area index, canopy height, surface roughness
- **Root depth**: affects moisture availability
- **Stomatal conductance**: differs between crops, forests, and urban surfaces
- **Season**: growth stage / phenological calendar (expressed here as monthly variation)

## How Kc is applied

### When the energy balance is active (`simulation.energy_balance = 1L`)

The $K_c$ factor is **applied to the turbulent exchange coefficient** $C_H$ before the energy balance solver is called:

$$C_H^{\text{adj}} = K_c \cdot C_H$$

!!! warning "Why not multiply PET by Kc?"
    Scaling PET by $K_c$ after the energy balance would keep $T_s$ solved with the original (wrong) $C_H$.  
    Because $H$ and $LE$ share the same $C_H$ in the energy budget, the correct $T_s$ depends on $K_c \cdot C_H$, not on the bare-soil value.

### When a precomputed PET raster is used, or in constant-PET mode

When the PET is obtained from the meteorological forcing data (raster NetCDF) or the energy balance is not active, $K_c$ is applied directly to PET:

$$PET_c = K_c \cdot \text{PET}_{\text{raster/constant}}$$

This is the standard FAO-56 formula applied as a simple multiplicative factor, which is the only option when PET is treated as an external input.

### When a MeteoRaster contains `et` and/or `pet` variables

| Raster contains | Behaviour |
|---|---|
| `et` | Used directly as actual evapotranspiration; $K_c$ is **not** applied again (it is assumed to be already embedded). The energy balance is skipped regardless of `simulation.energy_balance`. |
| `pet` | $K_c$ is applied ($PET_c = K_c \cdot \text{PET}$) before the soil water balance, and then $PET_c$ is used as input of the soil water balance module to calculate actual ET. The energy balance is skipped regardless of `simulation.energy_balance`. |
| both `et` and `pet` | `et` takes precedence and `pet` is ignored. |
| neither | Normal energy-balance or constant-PET path applies (see above). |

### Output variable naming (`et` vs `pet`)

The variable saved in `meteo_forcing.nc` and used as a state diagnostic changes name depending on whether $K_c$ was applied:

- **`et`** — actual crop evapotranspiration ($ET_c$): saved when $K_c \ne 1$ and the energy balance is active.
- **`pet`** — potential evapotranspiration (reference ET): saved when $K_c = 1$ (no land-cover adjustment).

The state grid (`output_states.evapotranspiration`) is always saved as **`ET`** [m/s] and contains the actual ET as bounded by soil water availability (output of the soil water balance module).

## Monthly Kc values

The default mapping is included with the package at `mobidic/data/kc_clc_mapping.csv`. The file contains one row per CLC class with 12 monthly $K_c$ values (January–December):

```csv
clc_code,kc_jan,kc_feb,...,kc_dec
111,0.10,0.10,...,0.10
311,0.60,0.60,...,0.60
312,1.00,1.00,...,1.00
...
```

Comment lines starting with `#` are ignored. The default values are derived from the FAO-56 guidelines adapted to the CLC classification.


## Configuration

### Enabling the CLC raster

Set the path to the Corine Land Cover GeoTIFF under `raster_files.CLC`:

```yaml
raster_files:
  CLC: example/raster/CLC_level3.tif   # OPTIONAL: CLC level 3 class codes
```

If the CLC raster is **not provided**, a uniform $K_c$ equal to `parameters.soil.Kc` is applied across the whole basin (default `1.0`).

### Default Kc

```yaml
parameters:
  soil:
    Kc: 1.0          # Scalar default Kc; used where CLC is absent or not in the mapping
```

### Custom Kc/CLC mapping

To override the default mapping shipped with the package, point `Kc_CLC_map` to a user-provided CSV with the same column layout:

```yaml
parameters:
  soil:
    Kc_CLC_map: custom/my_kc_mapping.csv   # OPTIONAL (blank = use built-in default)
```

The file must contain the columns `clc_code, kc_jan, kc_feb, kc_mar, kc_apr, kc_may, kc_jun, kc_jul, kc_aug, kc_sep, kc_oct, kc_nov, kc_dec`. CLC codes not present in the file fall back to `parameters.soil.Kc`.

### Complete example

```yaml
raster_files:
  CLC: geodat/CLC_level3.tif

parameters:
  soil:
    Kc: 1.0                           # Default for unclassified or missing codes
    Kc_CLC_map:                       # Leave blank to use the built-in default

simulation:
  energy_balance: 1L                  # Kc applied to CH when energy balance is active
```

## Implementation notes

- The Kc grid is rebuilt only when the **month changes** between timesteps (per-month cache).
- CLC is an integer classification raster. During grid decimation it is sub-sampled using **nearest-neighbour** (upper-left sub-cell) rather than averaging, to preserve class codes.
- The CLC grid is saved to and loaded from the consolidated `gisdata.nc` file (as the `CLC` variable) so that the mapping can be applied without the original GeoTIFF at simulation time.
- When the CLC raster is present but a cell's class code is not in the mapping, that cell gets `parameters.soil.Kc`.

## Functions

::: mobidic.core.crop_coefficients.load_kc_clc_mapping

::: mobidic.core.crop_coefficients.compute_kc_grid

::: mobidic.core.crop_coefficients.default_kc_clc_mapping_path