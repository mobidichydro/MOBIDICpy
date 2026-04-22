"""FAO single crop coefficient (Kc) support for actual evapotranspiration.

Maps Corine Land Cover (CLC) level 3 codes to monthly Kc values.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_MONTH_COLUMNS = [
    "kc_jan",
    "kc_feb",
    "kc_mar",
    "kc_apr",
    "kc_may",
    "kc_jun",
    "kc_jul",
    "kc_aug",
    "kc_sep",
    "kc_oct",
    "kc_nov",
    "kc_dec",
]


def default_kc_clc_mapping_path() -> Path:
    """Return the path to the CSV shipped with the package."""
    return Path(str(files("mobidic.data").joinpath("kc_clc_mapping.csv")))


def load_kc_clc_mapping(path: str | Path | None = None) -> dict[int, np.ndarray]:
    """Load CLC code -> monthly Kc mapping from CSV.

    The file may include comment lines starting with ``#``. It must contain a
    header row with the columns ``clc_code, kc_jan, kc_feb, ..., kc_dec``.

    Args:
        path: Path to a CSV file. If ``None``, the default mapping shipped with
            the package is used.

    Returns:
        Dictionary mapping integer CLC code to a length-12 array of monthly Kc
        values (January..December).
    """
    csv_path = Path(path) if path else default_kc_clc_mapping_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Kc/CLC mapping file not found: {csv_path}")

    df = pd.read_csv(csv_path, comment="#")
    expected = ["clc_code", *_MONTH_COLUMNS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Kc/CLC mapping file {csv_path} is missing columns: {missing}. Expected columns: {expected}")

    mapping: dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        code = int(row["clc_code"])
        values = np.asarray([row[c] for c in _MONTH_COLUMNS], dtype=float)
        mapping[code] = values

    logger.debug(f"Loaded Kc/CLC mapping from {csv_path}: {len(mapping)} classes")
    return mapping


def compute_kc_grid(
    clc_grid: np.ndarray | None,
    mapping: dict[int, np.ndarray],
    month: int,
    default_kc: float,
) -> np.ndarray | float:
    """Build a Kc grid for a given month from the CLC raster and mapping.

    Cells whose CLC code is not found in the mapping fall back to ``default_kc``.
    If ``clc_grid`` is ``None``, the scalar ``default_kc`` is returned so that
    callers can multiply PET without allocating an extra grid.

    Args:
        clc_grid: 2D grid of CLC level 3 codes (NaN outside domain). May be None.
        mapping: CLC code -> 12 monthly Kc values.
        month: Current month (1..12).
        default_kc: Kc value used where CLC is NaN or not present in the mapping.

    Returns:
        2D Kc grid with the same shape as ``clc_grid``, or ``default_kc`` as a
        scalar when ``clc_grid`` is None.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}")

    if clc_grid is None:
        return float(default_kc)

    kc = np.full_like(clc_grid, default_kc, dtype=float)
    finite = np.isfinite(clc_grid)
    if not finite.any():
        return kc

    codes = np.rint(clc_grid[finite]).astype(np.int64)
    month_idx = month - 1
    unique_codes = np.unique(codes)
    kc_finite = np.full(codes.shape, default_kc, dtype=float)
    for code in unique_codes:
        values = mapping.get(int(code))
        if values is None:
            continue
        kc_finite[codes == code] = values[month_idx]

    kc[finite] = kc_finite
    return kc
