"""Shared pytest configuration."""

import os
import tempfile

# netCDF4 >= 1.7.4 ships HDF5 filter plugins (blosc, zstd, ...) and points HDF5_PLUGIN_PATH at them when the
# variable is unset. HDF5 then probes every plugin on each Variable.filters() call (~40 ms on Windows), and xarray
# makes that call for every variable it opens, which made the NetCDF tests several times slower. MOBIDICpy only
# writes zlib-compressed files, and zlib is built into HDF5, so the tests run with an empty plugin directory.
# Must run before netCDF4 is imported; an HDF5_PLUGIN_PATH already set by the user is left unchanged.
if "HDF5_PLUGIN_PATH" not in os.environ:
    _no_plugins = os.path.join(tempfile.gettempdir(), "mobidicpy-tests-no-hdf5-plugins")
    os.makedirs(_no_plugins, exist_ok=True)
    os.environ["HDF5_PLUGIN_PATH"] = _no_plugins
