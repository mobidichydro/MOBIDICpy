"""Dynamic-state handling for sequential data assimilation.

"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mobidic.core.simulation import SimulationState

#: Grid state variables carried between cycles. ``wp``/``h``/``ts``/``td`` are
#: optional and only stored when present on the state.
GRID_VARIABLES = ("wc", "wg", "wp", "ws", "flr", "fld", "h", "ts", "td")

#: Per-reach state variables carried between cycles.
REACH_VARIABLES = ("discharge", "lateral_inflow")

#: Reservoir state fields carried between cycles.
RESERVOIR_FIELDS = ("volume", "stage", "inflow", "outflow", "withdrawal")

#: Grid variables that must always be present in a state file.
REQUIRED_GRID_VARIABLES = ("wc", "wg", "ws", "flr", "fld")

#: Largest state identifier; keeps the value inside a 12-character template field.
MAX_STATE_ID = 1_000_000

_STATE_FILE_RE = re.compile(r"^c(\d{4})_(\d{6})\.npz$")


# ---------------------------------------------------------------------------
# (a) State files
# ---------------------------------------------------------------------------


def state_file_path(directory: Path | str, cycle: int, state_id: int) -> Path:
    """Path of the state file written by ``cycle`` under identifier ``state_id``."""
    return Path(directory) / f"c{cycle:04d}_{state_id:06d}.npz"


def new_state_id(directory: Path | str, cycle: int) -> int:
    """Claim an unused state identifier for ``cycle``.

    The file is created empty and exclusively, so two concurrently running
    agents can never choose the same number.

    Args:
        directory: State-file directory.
        cycle: Cycle the state belongs to.

    Returns:
        An integer below :data:`MAX_STATE_ID`.

    Raises:
        RuntimeError: If no free identifier is found.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for _ in range(1000):
        candidate = random.randrange(MAX_STATE_ID)
        path = state_file_path(directory, cycle, candidate)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate

    raise RuntimeError(f"Could not claim a free state identifier in {directory} for cycle {cycle}")


def _pack_grid(name: str, array: np.ndarray, mask: np.ndarray, payload: dict) -> None:
    """Store a grid variable as active cells plus a uniform outside fill.

    Falls back to storing the full array when the values outside the mask are
    not all identical, so the round-trip is exact in every case.
    """
    array = np.asarray(array, dtype=np.float64)
    outside = array[~mask]
    if outside.size == 0:
        fill = np.float64(np.nan)
        uniform = True
    else:
        first = outside.flat[0]
        uniform = bool(np.all(np.isnan(outside))) if np.isnan(first) else bool(np.all(outside == first))
        fill = np.float64(first)

    if uniform:
        payload[f"grid__{name}__in"] = array[mask]
        payload[f"grid__{name}__fill"] = fill
    else:
        payload[f"grid__{name}__full"] = array


def _unpack_grid(name: str, payload, mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`_pack_grid`."""
    full_key = f"grid__{name}__full"
    if full_key in payload:
        return np.asarray(payload[full_key], dtype=np.float64).reshape(shape)

    values = np.asarray(payload[f"grid__{name}__in"], dtype=np.float64)
    fill = float(payload[f"grid__{name}__fill"])
    array = np.full(shape, fill, dtype=np.float64)
    array[mask] = values
    return array


def write_state_file(path: Path | str, state: "SimulationState", mask: np.ndarray) -> Path:
    """Write a full-resolution simulation state to a compressed ``.npz``.

    Only cells inside ``mask`` are stored for grid variables; the (uniform)
    value outside is stored as a single scalar. Everything is ``float64``: the
    cycle-chaining test compares a chained run against a continuous one at
    ``rtol=1e-10``.

    Args:
        path: Output file path.
        state: Simulation state to store.
        mask: 2D boolean array of active grid cells.

    Returns:
        The written path.
    """
    path = Path(path)
    mask = np.asarray(mask, dtype=bool)

    payload: dict[str, np.ndarray | np.float64] = {}
    present: list[str] = []
    for name in GRID_VARIABLES:
        array = getattr(state, name, None)
        if array is None:
            continue
        _pack_grid(name, array, mask, payload)
        present.append(name)

    missing = [name for name in REQUIRED_GRID_VARIABLES if name not in present]
    if missing:
        raise ValueError(
            f"Cannot write state file {path}: required state variable(s) {missing} are None. "
            "A state missing flr/fld would introduce a one-timestep discontinuity at every cycle boundary."
        )

    for name in REACH_VARIABLES:
        payload[f"reach__{name}"] = np.asarray(getattr(state, name), dtype=np.float64)

    reservoir_states = getattr(state, "reservoir_states", None)
    if reservoir_states:
        for field in RESERVOIR_FIELDS:
            payload[f"reservoir__{field}"] = np.array(
                [float(getattr(rs, field)) for rs in reservoir_states], dtype=np.float64
            )

    payload["meta__grid_variables"] = np.array(present, dtype="U16")
    payload["meta__shape"] = np.array(mask.shape, dtype=np.int64)
    payload["meta__n_active"] = np.array(int(mask.sum()), dtype=np.int64)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **payload)

    logger.debug(f"Wrote state file {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def read_state_file(path: Path | str, mask: np.ndarray, template: "SimulationState | None" = None) -> "SimulationState":
    """Read a state file written by :func:`write_state_file`.

    Args:
        path: State file path.
        mask: The same 2D boolean mask used when writing.
        template: Optional state of the same model setup, used only to supply
            the optional variables (``wp``, ``h``, ``ts``, ``td``) when they are
            absent from the file. They are absent only when the run that wrote
            the file did not have them, so the default of None is normally right.

    Returns:
        A new :class:`SimulationState`.

    Raises:
        FileNotFoundError: If the file does not exist. A missing state file is
            always an error: silently falling back to another state would carry
            the wrong state forward without any visible symptom.
        ValueError: If the file's grid shape or active-cell count does not match
            ``mask``.
    """
    from mobidic.core.simulation import SimulationState

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"State file not found: {path}. The previous cycle's state cannot be recovered; "
            "check that every PEST++ agent can read the shared state-file directory."
        )

    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape

    with np.load(path, allow_pickle=False) as payload:
        stored_shape = tuple(int(v) for v in payload["meta__shape"])
        if stored_shape != shape:
            raise ValueError(f"State file {path} has grid shape {stored_shape}, expected {shape}")

        # The mask decides which cells the packed values are scattered back into.
        # A mask that differs from the one used at write time silently relocates
        # every active cell; only a differing *count* raises on its own (via the
        # assignment below), and even that reports nothing about the cause.
        stored_active = int(payload["meta__n_active"])
        if stored_active != int(mask.sum()):
            raise ValueError(
                f"State file {path} was written with {stored_active} active grid cell(s) but the mask "
                f"supplied has {int(mask.sum())}. Both must come from build_state_mask() applied to the "
                "same gisdata; a different mask would scatter the stored values into the wrong cells."
            )

        present = [str(v) for v in payload["meta__grid_variables"]]
        grids = {name: _unpack_grid(name, payload, mask, shape) for name in present}
        reaches = {name: np.asarray(payload[f"reach__{name}"], dtype=np.float64) for name in REACH_VARIABLES}

        reservoir_values = None
        if "reservoir__volume" in payload:
            reservoir_values = {
                field: np.asarray(payload[f"reservoir__{field}"], dtype=np.float64) for field in RESERVOIR_FIELDS
            }

    missing = [name for name in REQUIRED_GRID_VARIABLES if name not in grids]
    if missing:
        raise ValueError(f"State file {path} is missing required state variable(s): {missing}")

    reservoir_states = None
    if reservoir_values is not None:
        from mobidic.core.reservoir import ReservoirState

        reservoir_states = [
            ReservoirState(**{field: float(reservoir_values[field][i]) for field in RESERVOIR_FIELDS})
            for i in range(len(reservoir_values["volume"]))
        ]

    def _grid(name):
        return grids.get(name, getattr(template, name, None))

    return SimulationState(
        wc=grids["wc"],
        wg=grids["wg"],
        wp=_grid("wp"),
        ws=grids["ws"],
        discharge=reaches["discharge"],
        lateral_inflow=reaches["lateral_inflow"],
        reservoir_states=reservoir_states,
        h=_grid("h"),
        ts=_grid("ts"),
        td=_grid("td"),
        flr=grids["flr"],
        fld=grids["fld"],
    )


def remove_old_state_files(directory: Path | str, cycle: int, keep: int = 2) -> int:
    """Delete state files older than ``cycle - keep + 1``.

    With ``keep = 2`` and a run at cycle ``c``, the files of cycle ``c - 1``
    (being read) and cycle ``c`` (being written) survive.

    Args:
        directory: State-file directory.
        cycle: Cycle currently being run.
        keep: Number of cycles of state files to retain.

    Returns:
        Number of files removed.
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    oldest_kept = cycle - keep + 1
    removed = 0
    for path in directory.glob("c*.npz"):
        match = _STATE_FILE_RE.match(path.name)
        if match is None:
            continue
        if int(match.group(1)) < oldest_kept:
            try:
                path.unlink()
                removed += 1
            except OSError:  # another agent may have removed it first
                continue

    if removed:
        logger.debug(f"Removed {removed} state file(s) older than cycle {oldest_kept} from {directory}")
    return removed


def build_state_mask(gisdata) -> np.ndarray:
    """Build the active-cell mask used for state files.

    A cell is active when it is inside the DTM domain or the flow-accumulation

    Args:
        gisdata: Loaded GISData with ``dtm`` and (optionally) ``flow_acc`` grids.

    Returns:
        2D boolean array.
    """
    grids = gisdata.grids
    dtm = np.asarray(grids["dtm"], dtype=np.float64)
    mask = np.isfinite(dtm)
    flow_acc = grids.get("flow_acc")
    if flow_acc is not None:
        mask = mask | np.isfinite(np.asarray(flow_acc, dtype=np.float64))
    return mask


# ---------------------------------------------------------------------------
# (b) The state identifier
# ---------------------------------------------------------------------------

#: Prefix of the PEST *parameter* names carrying a state quantity. Used to tell
#: interface parameters apart from MOBIDIC configuration parameters when reading
#: results back.
STATE_PAR_PREFIX = "sp_"

#: Prefix of the PEST *observation* names carrying a simulated state quantity.
STATE_OBS_PREFIX = "st_"

#: Reserved ``parameter_key`` used for the state identifier in ``model_input.csv``.
STATE_ID_INPUT_KEY = "__state_id__"

#: PEST parameter/observation name of the state identifier. This scalar is the
#: only state quantity that travels through the PEST interface: it names the
#: state file holding the full-resolution state, which PESTPP-DA then carries
#: from the accepted run of one cycle into the next.
STATE_ID_PAR_NAME = "sp_state_id"
STATE_ID_OBS_NAME = "st_state_id"


# ---------------------------------------------------------------------------
# (c) The assimilation space (formulation 2)
# ---------------------------------------------------------------------------

#: Reserved ``model_input.csv`` key prefix of an estimated state value.
STATE_INPUT_PREFIX = "__state__."

#: Estimable state variables.
#:
#: ``discharge`` is per reach and absolute [m3/s]. The other three are per
#: *zone*, and differ in what the zone reduction produces:
#:
#: - the two **soil stores** have a capacity, so the state is the dimensionless
#:   saturation ``theta = sum(W over zone) / sum(W0 over zone)``, bounded by
#:   construction and comparable between zones of very different capacity;
#: - **surface storage** has no capacity in MOBIDIC (``ws0`` is all zeros and is
#:   passed to the soil balance as ``None``), so the state is the zone's *mean
#:   depth* in metres, bounded from a reference run like discharge.
#:
#: Both are decoded the same way — one multiplier per zone applied to every cell
#: of the realization's own field — so within-zone structure always survives an
#: update. See :func:`zone_saturation`, :func:`zone_mean` and
#: :func:`rescale_zone_field`.
KIND_DISCHARGE = "discharge"
KIND_SOIL_CAPILLARY = "soil_capillary"
KIND_SOIL_GRAVITATIONAL = "soil_gravitational"
KIND_SURFACE_WATER = "surface_water"

#: Distributed *parameters* estimated per zone rather than model states.
#:
#: A state is carried by the model and evolves; these are properties the model
#: reads every timestep from ``Simulation.param_grids``. The distinction matters
#: for the ensemble: a storage grows by a factor of 25 during a storm, so the
#: absolute prior spread set at cycle 0 becomes negligible relative to it and the
#: Kalman gain collapses. A dimensionless parameter has no such scale, so its
#: spread stays meaningful for the whole event. They are therefore declared as
#: ordinary adjustable parameters (no ``state_par_link``): PESTPP-DA carries
#: their posterior ensemble from cycle to cycle by itself.
#:
#: ``runoff_fraction`` is MOBIDIC's ``f0``, distributed per zone as
#: ``mobidic_sid.m`` allows through ``f0file``. ``conductivity`` is a multiplier
#: on ``ks``. Both act directly on Hortonian runoff generation,
#: ``Rh = P*exp(-(1-f0)*ks*kaug/P)``.
KIND_RUNOFF_FRACTION = "runoff_fraction"
KIND_CONDUCTIVITY = "conductivity"

#: Short tag used inside PEST names, one per estimable quantity.
STATE_KIND_TAGS = {
    KIND_DISCHARGE: "q",
    KIND_SOIL_CAPILLARY: "wc",
    KIND_SOIL_GRAVITATIONAL: "wg",
    KIND_SURFACE_WATER: "ws",
    KIND_RUNOFF_FRACTION: "f0",
    KIND_CONDUCTIVITY: "ks",
}

#: ``Simulation.param_grids`` entry each zone parameter writes, and whether its
#: value replaces the grid outright or multiplies it.
ZONE_PARAM_GRID = {KIND_RUNOFF_FRACTION: "f0", KIND_CONDUCTIVITY: "ks"}
ZONE_PARAM_IS_MULTIPLIER = {KIND_RUNOFF_FRACTION: False, KIND_CONDUCTIVITY: True}

#: ``SimulationState`` grid each soil kind projects onto its zones. These are
#: normalised by a capacity; :data:`KIND_SURFACE_WATER` deliberately is not.
SOIL_KIND_FIELD = {KIND_SOIL_CAPILLARY: "wc", KIND_SOIL_GRAVITATIONAL: "wg"}

#: ``SimulationState`` grid of every zone-reduced *state* kind.
ZONE_KIND_FIELD = {**SOIL_KIND_FIELD, KIND_SURFACE_WATER: "ws"}

#: Every kind expressed per zone, states and parameters alike.
ZONAL_KINDS = frozenset(ZONE_KIND_FIELD) | frozenset(ZONE_PARAM_GRID)

#: ``da.states.estimate`` entries that stand for more than one kind.
ESTIMATE_ALIASES = {"soil_moisture": (KIND_SOIL_CAPILLARY, KIND_SOIL_GRAVITATIONAL)}

#: Cell value of the zone map outside every zone.
NO_ZONE = -1

#: Zone map written next to the state spec, so the forward model reconstructs
#: exactly the zones the setup built.
ZONE_MAP_FILE = "zone_map.npy"

#: Template field width of a state value. Wider than the identifier's 12: a
#: discharge needs room for both a peak and the digits that matter on a
#: recession, and PEST++ silently truncates to the field width.
STATE_VALUE_WIDTH = 20

#: Name of the JSON file describing the estimated states, written at setup and
#: read by the forward model.
STATE_SPEC_FILE = "da_state_spec.json"


def resolve_estimate_kinds(estimate) -> tuple[str, ...]:
    """Expand ``da.states.estimate`` into the state kinds it selects.

    ``soil_moisture`` is an alias for the two soil stores, which are estimated
    as separate states so the filter can move the capillary and gravitational
    storage independently.

    Args:
        estimate: Values of ``da.states.estimate``.

    Returns:
        Kinds in a stable order (discharge first, then the soil stores).

    Raises:
        ValueError: If an entry is unknown, or if two entries select the same kind.
    """
    kinds: list[str] = []
    for name in estimate:
        selected = ESTIMATE_ALIASES.get(name, (name,))
        for kind in selected:
            if kind not in STATE_KIND_TAGS:
                raise ValueError(
                    f"da.states.estimate: unknown state variable '{name}' "
                    f"(supported: {sorted(STATE_KIND_TAGS)} and {sorted(ESTIMATE_ALIASES)})"
                )
            if kind in kinds:
                raise ValueError(f"da.states.estimate selects '{kind}' more than once: {list(estimate)}")
            kinds.append(kind)
    order = list(STATE_KIND_TAGS)
    return tuple(sorted(kinds, key=order.index))


def state_par_name(kind: str, unit_id: int) -> str:
    """PEST *parameter* name of an estimated state (its value at the cycle start).

    ``unit_id`` is a reach ``mobidic_id`` for ``discharge`` and a zone id for the
    soil stores. With ``zones: reach`` a zone id *is* the ``mobidic_id`` of the
    reach the zone's cells drain to, which is what lets one localizer rule cover
    both.
    """
    return f"{STATE_PAR_PREFIX}{STATE_KIND_TAGS[kind]}_{unit_id:04d}"


def state_obs_name(kind: str, unit_id: int) -> str:
    """PEST *observation* name of an estimated state (its simulated value at the cycle end)."""
    return f"{STATE_OBS_PREFIX}{STATE_KIND_TAGS[kind]}_{unit_id:04d}"


def state_input_key(kind: str, unit_id: int) -> str:
    """Reserved ``model_input.csv`` key of an estimated state."""
    return f"{STATE_INPUT_PREFIX}{STATE_KIND_TAGS[kind]}.{unit_id:04d}"


def is_state_input_key(key: str) -> bool:
    """True for a ``model_input.csv`` row holding an estimated state value."""
    return key.startswith(STATE_INPUT_PREFIX)


def upstream_reaches(network, reach_ids) -> list[int]:
    """Return ``reach_ids`` plus every reach that drains into them.

    A gauge constrains only the reaches upstream of it; everything else in the
    network is independent of what it measures, so estimating those states adds
    nothing but spurious ensemble correlations.

    Args:
        network: Network GeoDataFrame with ``mobidic_id``, ``upstream_1`` and
            ``upstream_2`` columns.
        reach_ids: Reaches to start the traversal from.

    Returns:
        Sorted list of ``mobidic_id`` values in the upstream closure.

    Raises:
        KeyError: If a starting reach is not in the network.
    """
    ids = np.asarray(network["mobidic_id"].values, dtype=np.int64)
    position = {int(v): i for i, v in enumerate(ids)}
    upstream = [np.asarray(network[c].values, dtype=np.float64) for c in ("upstream_1", "upstream_2")]

    missing = [int(r) for r in reach_ids if int(r) not in position]
    if missing:
        raise KeyError(f"Reach(es) {missing} are not in the network")

    seen: set[int] = set()
    stack = [int(r) for r in reach_ids]
    while stack:
        reach = stack.pop()
        if reach in seen:
            continue
        seen.add(reach)
        row = position[reach]
        # 'no upstream reach' is NaN in the parquet, not a sentinel integer.
        for column in upstream:
            value = column[row]
            if np.isfinite(value) and value >= 0:
                stack.append(int(value))

    return sorted(seen)


# --- zones: the space a soil-moisture update is expressed in ----------------


def build_reach_zone_map(
    gisdata,
    network=None,
    reach_ids=None,
    min_zone_cells: int = 1,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build the soil-moisture zone map from the hillslope-reach mapping.

    ``gisdata.hillslope_reach_map`` already answers, for every cell, which reach
    it drains to. Taking that as the zone definition costs nothing and gives
    zones that are hydrologically meaningful: the cells of one zone all feed the
    same reach, so a gauge's information about that reach is information about
    exactly those cells.

    Zones are the space in which an *update* is expressed, never how the state
    is stored: the full-resolution field is always carried in the state file
    (see :func:`rescale_to_zone_saturation`).

    Args:
        gisdata: Loaded GISData with ``hillslope_reach_map`` and a ``dtm`` grid.
        network: Network GeoDataFrame (default: ``gisdata.network``). Used for
            the ``downstream`` column when merging small zones.
        reach_ids: Restrict the zones to these reaches (typically the upstream
            closure of the gauges). Cells draining anywhere else get no zone.
        min_zone_cells: Zones holding fewer than this many cells are merged into
            the nearest downstream zone that is large enough. A one-cell zone
            carries almost no information but still costs a parameter and adds a
            column the ensemble can correlate spuriously.

    Returns:
        Tuple ``(zone_map, zone_ids)``: a 2D integer array holding a zone id per
        cell (:data:`NO_ZONE` outside every zone), and the ascending ids of the
        zones that actually hold at least one cell.

    Raises:
        ValueError: If the hillslope map and the DTM disagree on shape, or if no
            zone survives.
    """
    grids = gisdata.grids
    dtm = np.asarray(grids["dtm"], dtype=np.float64)
    hillslope = np.asarray(gisdata.hillslope_reach_map)
    if hillslope.shape != dtm.shape:
        raise ValueError(
            f"hillslope_reach_map has shape {hillslope.shape} but the DTM has {dtm.shape}; "
            "both must come from the same preprocessing run"
        )

    # Unassigned cells are -9999 in the hillslope map; the soil water balance
    # itself runs on `isfinite(dtm) & (hillslope_reach_map >= 0)`, so the zones
    # must be exactly those cells and nothing else.
    numeric = hillslope.astype(np.float64)
    active = np.isfinite(dtm) & np.isfinite(numeric) & (numeric >= 0)
    zone_map = np.where(active, numeric, NO_ZONE).astype(np.int64)

    if reach_ids is not None:
        allowed = np.asarray(sorted({int(r) for r in reach_ids}), dtype=np.int64)
        zone_map = np.where(np.isin(zone_map, allowed), zone_map, NO_ZONE)

    if min_zone_cells > 1:
        if network is None:
            network = gisdata.network
        zone_map = _merge_small_zones(zone_map, network, min_zone_cells, reach_ids)

    zone_ids = tuple(int(v) for v in np.unique(zone_map) if v != NO_ZONE)
    if not zone_ids:
        raise ValueError(
            "No soil-moisture zone holds any cell. Check that the hillslope-reach mapping is "
            "present in gisdata and that da.states.estimate_reaches selects reaches that drain "
            "hillslope cells."
        )

    counts = np.bincount(zone_map[zone_map != NO_ZONE].ravel())
    sizes = counts[np.asarray(zone_ids, dtype=np.int64)]
    logger.info(
        f"Soil-moisture zones: {len(zone_ids)} zone(s) covering {int(sizes.sum())} active cell(s); "
        f"zone size {int(sizes.min())} to {int(sizes.max())} cells (median {int(np.median(sizes))})"
    )
    return zone_map, zone_ids


def _merge_small_zones(zone_map: np.ndarray, network, min_cells: int, allowed=None) -> np.ndarray:
    """Merge zones below ``min_cells`` into the nearest large downstream zone.

    Merging follows the drainage direction, so a merged zone stays a connected
    piece of the catchment. A chain of small zones running to the outlet has no
    large zone to merge into; those keep the last zone of the chain.
    """
    ids = np.asarray(network["mobidic_id"].values, dtype=np.int64)
    downstream_raw = np.asarray(network["downstream"].values, dtype=np.float64)
    downstream = {
        int(reach): int(value) for reach, value in zip(ids, downstream_raw) if np.isfinite(value) and value >= 0
    }
    permitted = None if allowed is None else {int(r) for r in allowed}

    present = zone_map[zone_map != NO_ZONE]
    if present.size == 0:
        return zone_map
    counts = np.bincount(present.ravel())

    def size(zone: int) -> int:
        return int(counts[zone]) if zone < counts.size else 0

    target: dict[int, int] = {}

    def resolve(zone: int) -> int:
        if zone in target:
            return target[zone]
        chain: list[int] = []
        current = zone
        seen: set[int] = set()
        while size(current) < min_cells and current not in seen:
            seen.add(current)
            chain.append(current)
            nxt = downstream.get(current)
            # Stop at the edge of the estimated set: merging past a gauge would
            # move cells into a zone the filter does not even estimate.
            if nxt is None or (permitted is not None and nxt not in permitted):
                break
            current = nxt
        for member in chain:
            target[member] = current
        target[zone] = current
        return current

    zones = [int(z) for z in np.unique(present)]
    mapping = np.arange(counts.size, dtype=np.int64)
    for zone in zones:
        mapping[zone] = resolve(zone)

    merged = np.where(zone_map != NO_ZONE, mapping[np.clip(zone_map, 0, counts.size - 1)], NO_ZONE)
    n_merged = sum(1 for zone in zones if mapping[zone] != zone)
    if n_merged:
        logger.info(
            f"Merged {n_merged} zone(s) smaller than {min_cells} cell(s) into their downstream neighbour "
            f"({len(zones)} -> {len(set(int(mapping[z]) for z in zones))} zones)"
        )
    return merged


def _zone_positions(zone_map: np.ndarray, ids) -> np.ndarray:
    """Map every cell to its position in ``ids``, or -1 when it belongs to no zone.

    ``ids`` must be sorted ascending, which every builder in this module
    guarantees.
    """
    ids_array = np.asarray(ids, dtype=np.int64)
    flat = np.asarray(zone_map, dtype=np.int64).ravel()
    if ids_array.size == 0:
        return np.full(flat.shape, -1, dtype=np.int64)
    position = np.clip(np.searchsorted(ids_array, flat), 0, ids_array.size - 1)
    return np.where(ids_array[position] == flat, position, -1)


def zone_saturation(values: np.ndarray, capacity: np.ndarray, zone_map: np.ndarray, ids) -> np.ndarray:
    """Project a soil-water grid onto zone-averaged saturation.

    ``theta_z = sum(W over zone) / sum(W0 over zone)``, i.e. the ratio of the
    zone's *total* storage to its *total* capacity. Using the ratio of sums
    rather than the mean of per-cell ratios keeps the reduction mass-consistent:
    the zone holds exactly ``theta_z`` of the water it could hold, whatever the
    spread of capacities inside it.

    Args:
        values: 2D soil-water grid [m] (``Wc`` or ``Wg``).
        capacity: 2D capacity grid [m] (``Wc0`` or ``Wg0``), already adjusted by
            the realization's own multipliers.
        zone_map: 2D zone id per cell.
        ids: Ascending zone ids.

    Returns:
        1D array of length ``len(ids)``, clipped to ``[0, 1]``.
    """
    flat_values = np.asarray(values, dtype=np.float64).ravel()
    flat_capacity = np.asarray(capacity, dtype=np.float64).ravel()
    position = _zone_positions(zone_map, ids)

    usable = (position >= 0) & np.isfinite(flat_values) & np.isfinite(flat_capacity) & (flat_capacity > 0.0)
    n_zones = len(ids)
    stored = np.bincount(position[usable], weights=flat_values[usable], minlength=n_zones)
    space = np.bincount(position[usable], weights=flat_capacity[usable], minlength=n_zones)

    with np.errstate(invalid="ignore", divide="ignore"):
        theta = np.where(space > 0.0, stored / np.where(space > 0.0, space, 1.0), 0.0)
    return np.clip(theta, 0.0, 1.0)


def zone_mean(values: np.ndarray, zone_map: np.ndarray, ids) -> np.ndarray:
    """Project a grid onto its zone means, with no capacity normalisation.

    Used for surface storage, which has no capacity in MOBIDIC (``ws0`` is all
    zeros and reaches the soil balance as ``None``), so the only meaningful
    zone reduction is the mean depth over the zone's cells. Cells are equal
    area, so this is the zone's water volume divided by its area.

    Args:
        values: 2D grid [m].
        zone_map: 2D zone id per cell.
        ids: Ascending zone ids.

    Returns:
        1D array of length ``len(ids)``, clipped at zero. Empty zones give 0.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    position = _zone_positions(zone_map, ids)
    usable = (position >= 0) & np.isfinite(flat)
    n_zones = len(ids)
    total = np.bincount(position[usable], weights=flat[usable], minlength=n_zones)
    count = np.bincount(position[usable], minlength=n_zones)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.where(count > 0, count, 1), 0.0)
    return np.maximum(mean, 0.0)


def rescale_zone_field(
    background: np.ndarray,
    zone_map: np.ndarray,
    ids,
    target: np.ndarray,
    capacity: np.ndarray | None = None,
) -> np.ndarray:
    """Write an analysed zone value back onto the full-resolution field.

    The realization's *own* field is multiplied by one factor per zone, so that
    the zone reaches the analysed value:

    .. code-block:: text

        factor_z = target_z / background_z
        field    = clip(field_background * factor_z, 0, capacity)

    instead of the zone average being spread uniformly over its cells. The
    pattern inside a zone — which the model spent the whole warm-up building —
    is therefore preserved exactly, and only its overall level moves.

    With ``capacity`` the target is a **saturation** and the field is clipped at
    capacity; without it the target is a **mean depth** and only the floor at
    zero applies.

    Two edge cases need a stated rule:

    - A zone that holds no water has no structure to preserve, so the target is
      applied uniformly (``W = theta_target * W0``, or the mean depth itself).
    - Where the clip at capacity binds, the achieved zone average falls short of
      the analysed value. That is accepted rather than redistributed to the
      remaining cells: redistributing would move water into cells the analysis
      said nothing about. Without a capacity the clip cannot bind, so the target
      is always reached exactly.

    Cells outside every zone keep their background value, NaN padding included.

    Args:
        background: 2D field inherited from the state file [m].
        zone_map: 2D zone id per cell.
        ids: Ascending zone ids.
        target: Analysed value per zone (saturation, or mean depth [m]).
        capacity: 2D capacity grid [m], or None when the quantity has none.

    Returns:
        A new 2D array; ``background`` is never modified.
    """
    shape = np.asarray(background).shape
    values = np.array(np.asarray(background, dtype=np.float64).ravel(), copy=True)
    position = _zone_positions(zone_map, ids)
    n_zones = len(ids)

    target = np.asarray(target, dtype=np.float64)
    if target.shape != (n_zones,):
        raise ValueError(f"Expected {n_zones} zone value(s), got {target.shape}")

    if capacity is None:
        target = np.maximum(target, 0.0)
        usable = (position >= 0) & np.isfinite(values)
        # Without a capacity the 'space' of a zone is simply its cell count, so
        # the target mean depth times the count is the water the zone should hold.
        space = np.bincount(position[usable], minlength=n_zones).astype(np.float64)
        per_cell = None
    else:
        target = np.clip(target, 0.0, 1.0)
        flat_capacity = np.asarray(capacity, dtype=np.float64).ravel()
        usable = (position >= 0) & np.isfinite(values) & np.isfinite(flat_capacity) & (flat_capacity > 0.0)
        space = np.bincount(position[usable], weights=flat_capacity[usable], minlength=n_zones)
        per_cell = flat_capacity

    stored = np.bincount(position[usable], weights=values[usable], minlength=n_zones)
    wanted = target * space

    dry = stored <= 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.where(dry, 0.0, wanted / np.where(dry, 1.0, stored))

    cells = np.flatnonzero(usable)
    zone_of = position[cells]
    uniform = target[zone_of] if per_cell is None else target[zone_of] * per_cell[cells]
    scaled = np.where(dry[zone_of], uniform, factor[zone_of] * values[cells])
    upper = np.inf if per_cell is None else per_cell[cells]
    values[cells] = np.clip(scaled, 0.0, upper)
    return values.reshape(shape)


def rescale_to_zone_saturation(
    background: np.ndarray,
    capacity: np.ndarray,
    zone_map: np.ndarray,
    ids,
    target: np.ndarray,
) -> np.ndarray:
    """Write an analysed zone *saturation* back onto the full-resolution field.

    Thin wrapper over :func:`rescale_zone_field` for the capacity-normalised
    (soil) case; see that function for the rules it applies.
    """
    return rescale_zone_field(background, zone_map, ids, target, capacity=capacity)


def soil_capacities(simulation) -> dict[str, np.ndarray]:
    """Capacity grid of each soil kind, taken from a built :class:`Simulation`.

    These are ``Wc0``/``Wg0`` *after* the ``Wc_factor``/``Wg_factor``
    multipliers, the minimum-storage floor and the gravitational/capillary
    transition have been applied, so they are the capacities the realization
    actually runs with. A saturation must be normalised by the same capacities
    it will later be decoded with, which is why this comes from the simulation
    rather than from gisdata.
    """
    return {KIND_SOIL_CAPILLARY: simulation.wc0, KIND_SOIL_GRAVITATIONAL: simulation.wg0}


def build_upstream_localizer(
    network,
    obs_reaches: dict[str, int],
    spec: "StateSpec",
    global_par_names: list[str],
):
    """Build a topology-based localization matrix for PESTPP-DA.

    A gauge can only carry information about the reaches that drain into it, so
    a discharge state is allowed to be updated by an observation group only when
    its reach lies in that group's upstream closure. Everything else in the
    network is independent of what the gauge measures, and any correlation the
    ensemble finds there is an artefact of the sample size.

    This is the upstream adjacency matrix of ``private/pestpp-da/localizer/
    upstream_adj.m``, with the diagonal included (a gauge informs its own reach).
    That script forms the full N x N transitive closure with Floyd-Warshall;
    only the rows of gauged reaches are ever needed, so each is taken directly
    with :func:`upstream_reaches` instead — the same closure in linear time.

    Rows are observation *group* names rather than individual observations,
    which PESTPP-DA expands to that group's non-zero-weight observations. This
    matters in sequential assimilation: the weight cycle table changes which
    observations are active from cycle to cycle, and a row naming an
    observation that is zero-weighted in the current cycle is an error.

    Soil-moisture zones follow the same rule, because a zone is identified by
    the reach its cells drain to: a gauge may update the soil state of a zone
    exactly when that zone drains into it.

    Args:
        network: Network GeoDataFrame.
        obs_reaches: Observation group name -> observed ``mobidic_id``.
        spec: The assimilation space.
        global_par_names: Adjustable parameters that are not states. They are
            basin-wide, so every group may update them. Every adjustable
            parameter must appear as a column: PESTPP-DA treats one that is
            missing as fixed and stops adjusting it.

    Returns:
        DataFrame of 0.0/1.0, indexed by observation group name with one column
        per parameter.
    """
    import pandas as pd

    columns = list(global_par_names) + spec.par_names
    matrix = pd.DataFrame(0.0, index=list(obs_reaches), columns=columns, dtype=float)
    matrix.loc[:, list(global_par_names)] = 1.0

    for group, reach_id in obs_reaches.items():
        upstream = set(upstream_reaches(network, [reach_id]))
        n_allowed = 0
        for block in spec.blocks:
            allowed = [state_par_name(block.kind, unit) for unit in block.ids if unit in upstream]
            if allowed:
                matrix.loc[group, allowed] = 1.0
            n_allowed += len(allowed)
            logger.info(f"Localizer: group '{group}' may update {len(allowed)} of {len(block)} {block.kind} state(s)")
        if not n_allowed:
            logger.warning(
                f"Localizer: observation group '{group}' (reach {reach_id}) has no estimated state "
                "upstream of it, so it can only update the non-state parameters"
            )

    return matrix


@dataclass(frozen=True)
class StateBlock:
    """One estimated state variable, over its own set of units.

    Attributes:
        kind: State variable (see :data:`STATE_KIND_TAGS`).
        ids: Unit id of every estimated state, ascending. A reach
            ``mobidic_id`` for ``discharge``, a zone id for the soil stores.
        positions: Index of each unit into the state's per-reach arrays. Only
            meaningful for ``discharge``; empty for a zone block, which is
            projected through the zone map instead.
        initial: Value each state takes at the start of cycle 0 (``parval1``).
        lower: Lower bound of each state parameter.
        upper: Upper bound of each state parameter.
    """

    kind: str
    ids: tuple[int, ...]
    positions: tuple[int, ...]
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    #: True for a dynamic state, which the model simulates and PESTPP-DA carries
    #: forward through ``state_par_link``. False for a distributed *parameter*,
    #: which has no simulated counterpart: it is an ordinary adjustable
    #: parameter whose ensemble PESTPP-DA carries between cycles by itself.
    linked: bool = True
    #: Applied as a multiplier on the model's own grid rather than replacing it.
    multiplier: bool = False

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def tag(self) -> str:
        """Short tag used inside PEST names."""
        return STATE_KIND_TAGS[self.kind]

    @property
    def is_soil(self) -> bool:
        """True for a zone-averaged soil-moisture block (normalised by a capacity)."""
        return self.kind in SOIL_KIND_FIELD

    @property
    def is_zonal(self) -> bool:
        """True for any block indexed by zone rather than by reach."""
        return self.kind in ZONAL_KINDS

    @property
    def is_parameter(self) -> bool:
        """True for a distributed parameter block (not a dynamic state)."""
        return self.kind in ZONE_PARAM_GRID

    @property
    def group(self) -> str:
        """PEST parameter/observation group name of this block."""
        return f"state_{self.tag}"

    @property
    def par_names(self) -> list[str]:
        """PEST parameter names, in interface order."""
        return [state_par_name(self.kind, unit) for unit in self.ids]

    @property
    def obs_names(self) -> list[str]:
        """PEST observation names, in interface order."""
        return [state_obs_name(self.kind, unit) for unit in self.ids]

    @property
    def input_keys(self) -> list[str]:
        """``model_input.csv`` keys, in interface order."""
        return [state_input_key(self.kind, unit) for unit in self.ids]


@dataclass(frozen=True)
class StateSpec:
    """The subset of the simulation state the filter is allowed to adjust.

    One :class:`StateBlock` per estimated state variable, concatenated in a
    fixed order that defines the interface: the same order is used for the
    template rows, the instruction file, ``model_input.csv`` and
    ``model_output.csv``.

    Attributes:
        blocks: The estimated state variables, in interface order.
        zone_map: 2D zone id per cell (:data:`NO_ZONE` outside every zone).
            Present exactly when a soil block is.
    """

    blocks: tuple[StateBlock, ...]
    zone_map: np.ndarray | None = None

    def __len__(self) -> int:
        return sum(len(block) for block in self.blocks)

    @property
    def kinds(self) -> tuple[str, ...]:
        """Estimated state variables, in interface order."""
        return tuple(block.kind for block in self.blocks)

    @property
    def soil_blocks(self) -> tuple[StateBlock, ...]:
        """The zone-averaged soil-moisture blocks."""
        return tuple(block for block in self.blocks if block.is_soil)

    @property
    def zonal_blocks(self) -> tuple[StateBlock, ...]:
        """Every block indexed by zone (soil moisture and surface storage)."""
        return tuple(block for block in self.blocks if block.is_zonal)

    def block(self, kind: str) -> StateBlock | None:
        """The block of ``kind``, or None when it is not estimated."""
        return next((block for block in self.blocks if block.kind == kind), None)

    @property
    def reach_ids(self) -> tuple[int, ...]:
        """Reaches carrying a discharge state (empty when discharge is not estimated)."""
        block = self.block(KIND_DISCHARGE)
        return block.ids if block is not None else ()

    @property
    def positions(self) -> tuple[int, ...]:
        """Network row of each reach carrying a discharge state."""
        block = self.block(KIND_DISCHARGE)
        return block.positions if block is not None else ()

    @property
    def par_names(self) -> list[str]:
        """PEST parameter names of every block, in interface order."""
        return [name for block in self.blocks for name in block.par_names]

    @property
    def linked_blocks(self) -> tuple[StateBlock, ...]:
        """Blocks that are dynamic states, i.e. carry a ``state_par_link``."""
        return tuple(block for block in self.blocks if block.linked)

    @property
    def parameter_blocks(self) -> tuple[StateBlock, ...]:
        """Blocks that are distributed parameters rather than states."""
        return tuple(block for block in self.blocks if block.is_parameter)

    @property
    def obs_names(self) -> list[str]:
        """PEST observation names, in interface order.

        Only the *linked* blocks appear: a distributed parameter has no
        simulated counterpart, so it gets no observation and no state link.
        """
        return [name for block in self.linked_blocks for name in block.obs_names]

    @property
    def input_keys(self) -> list[str]:
        """``model_input.csv`` keys of every block, in interface order."""
        return [key for block in self.blocks for key in block.input_keys]

    @property
    def initial(self) -> np.ndarray:
        """``parval1`` of every state, in interface order."""
        return self._concat("initial")

    @property
    def lower(self) -> np.ndarray:
        """Lower bound of every state, in interface order."""
        return self._concat("lower")

    @property
    def upper(self) -> np.ndarray:
        """Upper bound of every state, in interface order."""
        return self._concat("upper")

    def _concat(self, field: str) -> np.ndarray:
        arrays = [np.asarray(getattr(block, field), dtype=np.float64) for block in self.blocks]
        return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float64)

    @classmethod
    def combine(cls, *specs: "StateSpec") -> "StateSpec":
        """Concatenate several single-variable specs into one assimilation space.

        Raises:
            ValueError: If two specs estimate the same kind, or carry different
                zone maps.
        """
        blocks: list[StateBlock] = []
        zone_map = None
        for spec in specs:
            for block in spec.blocks:
                if any(block.kind == existing.kind for existing in blocks):
                    raise ValueError(f"State kind '{block.kind}' appears in more than one spec")
                blocks.append(block)
            if spec.zone_map is None:
                continue
            if zone_map is None:
                zone_map = spec.zone_map
            elif not np.array_equal(zone_map, spec.zone_map):
                raise ValueError("Cannot combine specs built with different zone maps")
        return cls(blocks=tuple(blocks), zone_map=zone_map)

    def to_json(self, path: Path | str) -> Path:
        """Write the spec the forward model needs to reproduce the interface order.

        The zone map is written alongside as :data:`ZONE_MAP_FILE`, so the
        forward model reconstructs exactly the zones the setup built rather than
        recomputing them (a recomputed map could differ after any change to the
        preprocessing and would silently relocate every zone).
        """
        path = Path(path)
        payload = {
            "blocks": [
                {
                    "kind": block.kind,
                    "ids": [int(v) for v in block.ids],
                    "positions": [int(p) for p in block.positions],
                    "lower": [float(v) for v in block.lower],
                    "upper": [float(v) for v in block.upper],
                    "linked": bool(block.linked),
                    "multiplier": bool(block.multiplier),
                }
                for block in self.blocks
            ],
            "zone_map": ZONE_MAP_FILE if self.zone_map is not None else None,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        if self.zone_map is not None:
            np.save(path.parent / ZONE_MAP_FILE, np.asarray(self.zone_map, dtype=np.int32))
        return path

    @classmethod
    def from_json(cls, path: Path | str) -> "StateSpec":
        """Read a spec written by :meth:`to_json`."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        blocks = []
        for entry in payload["blocks"]:
            lower = np.asarray(entry["lower"], dtype=np.float64)
            blocks.append(
                StateBlock(
                    kind=entry["kind"],
                    ids=tuple(int(v) for v in entry["ids"]),
                    positions=tuple(int(p) for p in entry["positions"]),
                    initial=np.zeros_like(lower),  # only setup needs parval1
                    lower=lower,
                    upper=np.asarray(entry["upper"], dtype=np.float64),
                    linked=bool(entry.get("linked", True)),
                    multiplier=bool(entry.get("multiplier", False)),
                )
            )

        zone_map = None
        zone_file = payload.get("zone_map")
        if zone_file:
            zone_path = path.parent / zone_file
            if not zone_path.exists():
                raise FileNotFoundError(
                    f"The state spec {path} declares a zone map but {zone_path} is missing. "
                    "Every estimated soil state would be applied to the wrong cells."
                )
            zone_map = np.load(zone_path).astype(np.int64)

        return cls(blocks=tuple(blocks), zone_map=zone_map)

    def check_against(self, network) -> None:
        """Verify that the spec still describes the network it was built for.

        The forward model reloads the network from disk. If its row order ever
        differed from the order at setup time, every state would be written to
        the wrong reach — silently, and in a way no mass-balance check would
        catch.

        Raises:
            ValueError: If the network does not match the spec.
        """
        ids = np.asarray(network["mobidic_id"].values, dtype=np.int64)

        discharge = self.block(KIND_DISCHARGE)
        if discharge is not None:
            positions = np.asarray(discharge.positions, dtype=np.int64)
            if positions.size and positions.max() >= ids.size:
                raise ValueError(
                    f"State spec references reach position {int(positions.max())} but the network has "
                    f"only {ids.size} reaches"
                )
            if not np.array_equal(ids[positions], np.asarray(discharge.ids, dtype=np.int64)):
                raise ValueError(
                    "The network row order does not match the one the state spec was built with; "
                    "estimated states would be applied to the wrong reaches. Re-run the setup."
                )

        if self.zonal_blocks and self.zone_map is None:
            kinds = ", ".join(block.kind for block in self.zonal_blocks)
            raise ValueError(f"The state spec estimates zone-reduced state(s) ({kinds}) but carries no zone map")

    def check_grid(self, shape: tuple[int, int]) -> None:
        """Verify that the zone map matches the model grid.

        Raises:
            ValueError: If the zone map has a different shape.
        """
        if self.zone_map is None:
            return
        if tuple(np.asarray(self.zone_map).shape) != tuple(shape):
            raise ValueError(
                f"The zone map has shape {np.asarray(self.zone_map).shape} but the model grid is "
                f"{tuple(shape)}; the state spec was built for a different setup"
            )


def build_discharge_state_spec(
    network,
    initial_discharge: np.ndarray,
    reference_discharge: np.ndarray,
    reach_ids=None,
    bound_factor: float = 10.0,
    state_floor: float = 1.0e-30,
) -> StateSpec:
    """Build the discharge assimilation space.

    Args:
        network: Network GeoDataFrame.
        initial_discharge: Per-reach discharge at the start of cycle 0 [m3/s],
            in network row order. Used as ``parval1``.
        reference_discharge: Per-reach maximum discharge over the whole
            assimilation period from a deterministic reference run [m3/s]. The
            upper bounds are derived from it rather than from the initial value,
            because bounds are enforced on the state carried between cycles: an
            upper bound below the flow the model actually produces would
            truncate the transfer instead of merely limiting the update.
        reach_ids: Reaches to estimate (default: every reach in the network).
        bound_factor: Upper bound as a multiple of ``reference_discharge``.
        state_floor: Lower bound of every state [m3/s]. Must be positive: a lower
            bound of zero lets the ensemble update produce a *subnormal* value,
            which PEST++ refuses to run (``add_runs() error: denormal values``).
            Since the routing recursion decays exponentially, a reach that goes
            dry slides into the subnormal range and stays there. With a positive
            floor, bounds enforcement clips it back to a normal number.

    Returns:
        A :class:`StateSpec` in ascending ``mobidic_id`` order.

    Raises:
        ValueError: If the arrays do not match the network, or no reach is left.
    """
    ids = np.asarray(network["mobidic_id"].values, dtype=np.int64)
    initial_discharge = np.asarray(initial_discharge, dtype=np.float64)
    reference_discharge = np.asarray(reference_discharge, dtype=np.float64)
    for name, array in (("initial_discharge", initial_discharge), ("reference_discharge", reference_discharge)):
        if array.shape != ids.shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {ids.shape} (one value per reach)")

    selected = sorted(int(r) for r in (ids if reach_ids is None else reach_ids))
    if not selected:
        raise ValueError("No reach selected for discharge state estimation")
    if state_floor <= 0.0:
        raise ValueError(f"state_floor must be positive, got {state_floor}")

    position = {int(v): i for i, v in enumerate(ids)}
    positions = np.array([position[r] for r in selected], dtype=np.int64)

    # parval1 must not sit below parlbnd, so a dry reach starts on the floor.
    initial = np.maximum(initial_discharge[positions], state_floor)
    # A reach that carries no water at any point in the reference run still needs
    # a non-degenerate interval: PEST requires parubnd > parlbnd.
    upper = np.maximum(bound_factor * np.maximum(reference_discharge[positions], initial), state_floor * 1.0e3)

    logger.info(
        f"Discharge assimilation space: {len(selected)} of {ids.size} reaches, bounds "
        f"[{state_floor:.3g}, {upper.min():.3g}] to [{state_floor:.3g}, {upper.max():.3g}] m3/s "
        f"({bound_factor:g}x the reference peak)"
    )
    block = StateBlock(
        kind=KIND_DISCHARGE,
        ids=tuple(selected),
        positions=tuple(int(p) for p in positions),
        initial=initial,
        lower=np.full_like(upper, state_floor),
        upper=upper,
    )
    return StateSpec(blocks=(block,))


def build_soil_state_spec(
    kinds,
    state: "SimulationState",
    capacities: dict[str, np.ndarray],
    zone_map: np.ndarray,
    zone_ids,
    saturation_bounds: tuple[float, float] = (0.0, 1.0),
    state_floor: float = 1.0e-30,
) -> StateSpec:
    """Build the zone-averaged soil-moisture assimilation space.

    Unlike discharge, no reference run is needed to bound these states: a
    saturation is confined to ``[0, 1]`` by definition, so the bounds are
    physical rather than estimated, and PESTPP-DA can never truncate the
    transferred state by enforcing them.

    Args:
        kinds: Soil kinds to estimate (:data:`KIND_SOIL_CAPILLARY` and/or
            :data:`KIND_SOIL_GRAVITATIONAL`).
        state: State the cycle-0 values are read from (the warm-up final state).
        capacities: Capacity grid per kind, from :func:`soil_capacities`.
        zone_map: 2D zone id per cell.
        zone_ids: Ascending zone ids.
        saturation_bounds: Bounds on a zone saturation.
        state_floor: Positive floor applied to the lower bound. A lower bound of
            exactly zero leaves subnormal doubles inside the feasible interval,
            and PEST++ refuses to queue a run whose parameter vector holds one.

    Returns:
        A :class:`StateSpec` with one block per kind, sharing ``zone_map``.

    Raises:
        ValueError: If a kind is not a soil store, if a capacity grid is
            missing, or if the bounds are degenerate.
    """
    lower_bound = max(float(saturation_bounds[0]), float(state_floor))
    upper_bound = float(saturation_bounds[1])
    if not upper_bound > lower_bound:
        raise ValueError(f"saturation_bounds {saturation_bounds} do not leave a usable interval")
    if state_floor <= 0.0:
        raise ValueError(f"state_floor must be positive, got {state_floor}")

    zone_ids = tuple(int(z) for z in zone_ids)
    n_zones = len(zone_ids)
    blocks = []
    for kind in kinds:
        field = SOIL_KIND_FIELD.get(kind)
        if field is None:
            raise ValueError(f"'{kind}' is not a soil state kind (expected one of {sorted(SOIL_KIND_FIELD)})")
        grid = getattr(state, field, None)
        if grid is None:
            raise ValueError(f"The state carries no '{field}' grid, so '{kind}' cannot be estimated")
        capacity = capacities.get(kind)
        if capacity is None:
            raise ValueError(f"No capacity grid supplied for '{kind}'")

        theta = zone_saturation(grid, capacity, zone_map, zone_ids)
        initial = np.clip(theta, lower_bound, upper_bound)
        blocks.append(
            StateBlock(
                kind=kind,
                ids=zone_ids,
                positions=(),
                initial=initial,
                lower=np.full(n_zones, lower_bound, dtype=np.float64),
                upper=np.full(n_zones, upper_bound, dtype=np.float64),
            )
        )
        logger.info(
            f"Soil assimilation space '{kind}': {n_zones} zone(s), saturation bounds "
            f"[{lower_bound:.3g}, {upper_bound:g}], warm-up saturation {theta.min():.3f} to "
            f"{theta.max():.3f} (mean {theta.mean():.3f})"
        )

    return StateSpec(blocks=tuple(blocks), zone_map=np.asarray(zone_map, dtype=np.int64))


def build_surface_state_spec(
    state: "SimulationState",
    reference_surface: np.ndarray,
    zone_map: np.ndarray,
    zone_ids,
    bound_factor: float = 10.0,
    state_floor: float = 1.0e-30,
) -> StateSpec:
    """Build the zone-averaged surface-storage assimilation space.

    Surface storage has no capacity in MOBIDIC, so unlike the soil stores its
    state is an absolute mean depth [m] and its bounds cannot be physical. They
    come from a deterministic reference run instead, exactly as the discharge
    bounds do: PESTPP-DA enforces bounds on the state it *transfers* between
    cycles, so a bound sized from the (nearly empty) warm-up surface store would
    truncate the transfer once the storm fills it.

    Args:
        state: State the cycle-0 values are read from (the warm-up final state).
        reference_surface: Per-zone maximum mean depth over the whole
            assimilation period from a deterministic reference run [m].
        zone_map: 2D zone id per cell.
        zone_ids: Ascending zone ids.
        bound_factor: Upper bound as a multiple of ``reference_surface``.
        state_floor: Positive lower bound [m]. Must be positive for the same
            denormal reason as the discharge floor.

    Returns:
        A :class:`StateSpec` with a single surface-water block.

    Raises:
        ValueError: If the state carries no ``ws`` grid, if the reference does
            not match the zones, or if ``state_floor`` is not positive.
    """
    grid = getattr(state, "ws", None)
    if grid is None:
        raise ValueError("The state carries no 'ws' grid, so surface storage cannot be estimated")
    if state_floor <= 0.0:
        raise ValueError(f"state_floor must be positive, got {state_floor}")

    zone_ids = tuple(int(z) for z in zone_ids)
    reference = np.asarray(reference_surface, dtype=np.float64)
    if reference.shape != (len(zone_ids),):
        raise ValueError(f"reference_surface has shape {reference.shape}, expected {(len(zone_ids),)} (one per zone)")

    initial = np.maximum(zone_mean(grid, zone_map, zone_ids), state_floor)
    # A zone that never holds surface water still needs parubnd > parlbnd.
    upper = np.maximum(bound_factor * np.maximum(reference, initial), state_floor * 1.0e3)

    logger.info(
        f"Surface-storage assimilation space: {len(zone_ids)} zone(s), bounds [{state_floor:.3g}, "
        f"{upper.min():.3g}] to [{state_floor:.3g}, {upper.max():.3g}] m ({bound_factor:g}x the "
        f"reference maximum), warm-up mean depth {initial.min():.3g} to {initial.max():.3g} m"
    )
    block = StateBlock(
        kind=KIND_SURFACE_WATER,
        ids=zone_ids,
        positions=(),
        initial=initial,
        lower=np.full(len(zone_ids), state_floor, dtype=np.float64),
        upper=upper,
    )
    return StateSpec(blocks=(block,), zone_map=np.asarray(zone_map, dtype=np.int64))


def build_zone_parameter_spec(
    kinds,
    zone_map: np.ndarray,
    zone_ids,
    initial: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> StateSpec:
    """Build the distributed-parameter assimilation space (``f0``, ``ks``).

    These are not model states: they are properties the model reads from
    ``Simulation.param_grids`` every timestep. Being dimensionless they are
    scale-free, which is the point — a storage grows 25x during a storm and its
    absolute ensemble spread becomes negligible, whereas a bounded parameter's
    spread stays meaningful for the whole event.

    Args:
        kinds: Zone-parameter kinds (:data:`KIND_RUNOFF_FRACTION`,
            :data:`KIND_CONDUCTIVITY`).
        zone_map: 2D zone id per cell.
        zone_ids: Ascending zone ids.
        initial: Value each kind starts from (default: the model's own, i.e.
            ``f0`` from the timestep formula and 1.0 for the ``ks`` multiplier).
        bounds: ``(lower, upper)`` per kind.

    Returns:
        A :class:`StateSpec` with one unlinked block per kind.

    Raises:
        ValueError: If a kind is not a zone parameter, or its bounds are degenerate.
    """
    defaults_initial = {KIND_RUNOFF_FRACTION: 0.0187, KIND_CONDUCTIVITY: 1.0}
    defaults_bounds = {KIND_RUNOFF_FRACTION: (1.0e-6, 0.95), KIND_CONDUCTIVITY: (0.1, 10.0)}
    initial = {**defaults_initial, **(initial or {})}
    bounds = {**defaults_bounds, **(bounds or {})}

    zone_ids = tuple(int(z) for z in zone_ids)
    n = len(zone_ids)
    blocks = []
    for kind in kinds:
        if kind not in ZONE_PARAM_GRID:
            raise ValueError(f"'{kind}' is not a zone parameter (expected one of {sorted(ZONE_PARAM_GRID)})")
        lower, upper = (float(v) for v in bounds[kind])
        if not upper > lower > 0.0:
            raise ValueError(f"bounds for '{kind}' must satisfy 0 < lower < upper, got ({lower}, {upper})")
        value = float(np.clip(initial[kind], lower, upper))
        blocks.append(
            StateBlock(
                kind=kind,
                ids=zone_ids,
                positions=(),
                initial=np.full(n, value, dtype=np.float64),
                lower=np.full(n, lower, dtype=np.float64),
                upper=np.full(n, upper, dtype=np.float64),
                linked=False,
                multiplier=ZONE_PARAM_IS_MULTIPLIER[kind],
            )
        )
        logger.info(
            f"Zone-parameter space '{kind}': {n} zone(s), value {value:g} in [{lower:g}, {upper:g}] "
            f"({'multiplier on' if ZONE_PARAM_IS_MULTIPLIER[kind] else 'replaces'} "
            f"param_grids['{ZONE_PARAM_GRID[kind]}'])"
        )

    return StateSpec(blocks=tuple(blocks), zone_map=np.asarray(zone_map, dtype=np.int64))


def apply_zone_parameters(simulation, spec: "StateSpec", values: np.ndarray) -> int:
    """Write the estimated zone parameters into a built ``Simulation``.

    ``_prepare_grids()`` runs in ``Simulation.__init__`` and the main loop reads
    ``self.param_grids[...]`` every timestep, so overwriting those grids after
    construction is all that a distributed parameter needs — no change to the
    simulation engine. This is the same hook ``mobidic_sid.m`` exposes through
    ``f0file``.

    Cells outside every zone keep the model's own value, so a partial zone
    covering (``estimate_reaches: upstream``) leaves the rest of the basin alone.

    Args:
        simulation: A constructed :class:`Simulation`.
        spec: Assimilation space; only its parameter blocks are used.
        values: Full interface vector, in interface order.

    Returns:
        Number of parameter blocks applied.
    """
    applied = 0
    for block, start, stop in block_slices(spec):
        if not block.is_parameter:
            continue
        grid_name = ZONE_PARAM_GRID[block.kind]
        base = np.asarray(simulation.param_grids[grid_name], dtype=np.float64)
        per_zone = np.clip(np.asarray(values[start:stop], dtype=np.float64), block.lower, block.upper)

        position = _zone_positions(spec.zone_map, block.ids)
        inside = position >= 0
        flat = base.ravel().copy()
        selected = per_zone[position[inside]]
        flat[inside] = flat[inside] * selected if block.multiplier else selected
        simulation.param_grids[grid_name] = flat.reshape(base.shape)

        applied += 1
        logger.info(
            f"Applied {len(block)} '{block.kind}' zone value(s) to param_grids['{grid_name}'] "
            f"({int(inside.sum())} cells; range {per_zone.min():.4g} to {per_zone.max():.4g})"
        )
    return applied


def block_slices(spec: "StateSpec"):
    """Yield ``(block, start, stop)`` for each block's slice of the interface vector."""
    offset = 0
    for block in spec.blocks:
        yield block, offset, offset + len(block)
        offset += len(block)


def _require_capacity(capacities: dict[str, np.ndarray] | None, kind: str) -> np.ndarray:
    """Capacity grid of ``kind``, or a clear error explaining what is missing."""
    capacity = (capacities or {}).get(kind)
    if capacity is None:
        raise ValueError(
            f"Estimating '{kind}' needs its capacity grid: pass capacities=soil_capacities(simulation). "
            "A zone saturation is only defined relative to the capacities the realization runs with."
        )
    return capacity


def extract_state_vector(
    state: "SimulationState",
    spec: StateSpec,
    capacities: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Project a full model state onto the state vector, in interface order.

    Applied to a cycle's ``final_state`` this yields the *forecast* state vector
    that PESTPP-DA transfers into the next cycle's analysis.

    Args:
        state: Model state to project (normally a run's ``final_state``).
        spec: Assimilation space.
        capacities: Capacity grid per soil kind, from :func:`soil_capacities`.
            Required when the spec estimates soil moisture.

    Returns:
        1D array of length ``len(spec)``.
    """
    parts = []
    for block in spec.blocks:
        if block.is_parameter:
            # A distributed parameter has no simulated counterpart to report:
            # PESTPP-DA carries its ensemble itself, without state_par_link.
            continue
        if block.kind == KIND_DISCHARGE:
            parts.append(np.asarray(state.discharge, dtype=np.float64)[list(block.positions)])
        elif block.is_soil:
            grid = getattr(state, SOIL_KIND_FIELD[block.kind])
            parts.append(zone_saturation(grid, _require_capacity(capacities, block.kind), spec.zone_map, block.ids))
        else:
            # Surface storage has no capacity, so the reduction is a mean depth.
            grid = getattr(state, ZONE_KIND_FIELD[block.kind])
            parts.append(zone_mean(grid, spec.zone_map, block.ids))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def insert_state_vector(
    values: np.ndarray,
    spec: StateSpec,
    state: "SimulationState",
    capacities: dict[str, np.ndarray] | None = None,
) -> "SimulationState":
    """Insert an analysed state vector back into the full model state.

    The inverse of :func:`extract_state_vector`. Everything outside the
    assimilation space keeps the value the state file carried, so the analysis is
    an increment on the background state rather than a replacement of it: a
    reach with no discharge state keeps its simulated discharge, and a cell
    outside every zone keeps its simulated soil water.

    Args:
        values: Analysed state vector, in interface order.
        spec: Assimilation space.
        state: Background state, read from the previous cycle's state file.
            Modified in place; each adjusted array is replaced rather than
            written through, so the caller's arrays are never touched.
        capacities: Capacity grid per soil kind, from :func:`soil_capacities`.
            Required when the spec estimates soil moisture.

    Returns:
        The same state object.

    Raises:
        ValueError: If ``values`` does not match the spec length.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(spec),):
        raise ValueError(f"Expected {len(spec)} state value(s), got {values.shape}")

    for block, offset, stop in block_slices(spec):
        if block.is_parameter:
            # Not a state: applied to the model's parameter grids instead, by
            # apply_zone_parameters().
            continue
        block_values = values[offset:stop]

        if block.kind == KIND_DISCHARGE:
            negative = int((block_values < 0.0).sum())
            if negative:
                # da_enforce_bounds should have prevented this; a negative discharge
                # would propagate through the routing as a physical impossibility.
                logger.warning(f"{negative} analysed discharge state(s) were negative and have been clipped to zero")
            discharge = np.array(state.discharge, dtype=np.float64, copy=True)
            discharge[list(block.positions)] = np.maximum(block_values, 0.0)
            state.discharge = discharge
            continue

        field = ZONE_KIND_FIELD[block.kind]
        capacity = _require_capacity(capacities, block.kind) if block.is_soil else None
        updated = rescale_zone_field(
            background=getattr(state, field),
            zone_map=spec.zone_map,
            ids=block.ids,
            target=block_values,
            capacity=capacity,
        )
        setattr(state, field, updated)

        if capacity is not None:
            achieved = zone_saturation(updated, capacity, spec.zone_map, block.ids)
            shortfall = np.abs(achieved - np.clip(block_values, 0.0, 1.0))
            if shortfall.max() > 1.0e-6:
                # Only the clip at capacity can cause this (see the docstring).
                logger.debug(
                    f"{int((shortfall > 1.0e-6).sum())} '{block.kind}' zone(s) could not reach the analysed "
                    f"saturation because the field clipped at capacity (largest shortfall {shortfall.max():.3g})"
                )

    return state
