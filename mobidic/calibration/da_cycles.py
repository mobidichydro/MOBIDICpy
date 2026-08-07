"""Cycle scheduling and cycle-table generation for PESTPP-DA.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mobidic.calibration.config import ObservationGroup
from mobidic.calibration.observation import observation_weights

#: Name of the fixed parameter carrying the cycle number to the forward model.
CYCLE_PARAM_NAME = "cycle_num"

#: Reserved ``parameter_key`` used for the cycle number in ``model_input.csv``.
CYCLE_INPUT_KEY = "__cycle__"

#: Static CSV mapping cycle number to its absolute time window.
CYCLE_METADATA_FILE = "da_cycles.csv"


@dataclass(frozen=True)
class CycleSchedule:
    """Absolute time windows of every assimilation cycle.

    Attributes:
        starts: ``t_c`` for each cycle (inclusive).
        ends: ``t_{c+1} - dt`` for each cycle (inclusive).
        n_steps_per_cycle: Number of simulation timesteps in every cycle.
        dt_seconds: Simulation timestep [s].
    """

    starts: list[pd.Timestamp]
    ends: list[pd.Timestamp]
    n_steps_per_cycle: int
    dt_seconds: int

    @property
    def n_cycles(self) -> int:
        """Number of cycles in the schedule."""
        return len(self.starts)

    def slot_times(self, cycle: int) -> pd.DatetimeIndex:
        """Absolute times of every within-cycle observation slot of ``cycle``."""
        return pd.date_range(
            start=self.starts[cycle],
            periods=self.n_steps_per_cycle,
            freq=f"{self.dt_seconds}s",
        )

    def all_times(self) -> pd.DatetimeIndex:
        """Absolute times of every slot of every cycle, in order."""
        return pd.date_range(
            start=self.starts[0],
            periods=self.n_cycles * self.n_steps_per_cycle,
            freq=f"{self.dt_seconds}s",
        )


def build_cycle_schedule(
    sim_start: str | pd.Timestamp,
    sim_end: str | pd.Timestamp,
    cycle_length: str,
    dt_seconds: int,
) -> CycleSchedule:
    """Split ``[sim_start, sim_end]`` into uniform assimilation cycles.

    Args:
        sim_start: First simulation time (inclusive).
        sim_end: Last simulation time (inclusive).
        cycle_length: Cycle length as a pandas offset string (e.g. ``"6h"``).
        dt_seconds: Simulation timestep [s].

    Returns:
        CycleSchedule with one entry per complete cycle.

    Raises:
        ValueError: If ``cycle_length`` is not a positive multiple of the
            timestep, or if the window does not hold a single complete cycle.
    """
    start = pd.Timestamp(sim_start)
    end = pd.Timestamp(sim_end)
    if end < start:
        raise ValueError(f"Simulation end ({end}) is before simulation start ({start})")

    cycle_seconds = pd.Timedelta(cycle_length).total_seconds()
    if cycle_seconds <= 0:
        raise ValueError(f"cycle_length '{cycle_length}' must be positive")
    if cycle_seconds % dt_seconds != 0:
        raise ValueError(
            f"cycle_length '{cycle_length}' ({cycle_seconds:.0f}s) is not an integer multiple "
            f"of the simulation timestep ({dt_seconds}s)"
        )

    n_steps_per_cycle = int(cycle_seconds // dt_seconds)

    # The simulation window is inclusive of both endpoints.
    total_steps = int((end - start).total_seconds() // dt_seconds) + 1
    n_cycles = total_steps // n_steps_per_cycle
    if n_cycles < 1:
        raise ValueError(
            f"Simulation period [{start}, {end}] holds {total_steps} timestep(s), fewer than the "
            f"{n_steps_per_cycle} needed for one cycle of length '{cycle_length}'"
        )

    leftover = total_steps - n_cycles * n_steps_per_cycle
    if leftover:
        last_used = start + pd.Timedelta(seconds=(n_cycles * n_steps_per_cycle - 1) * dt_seconds)
        logger.warning(
            f"Simulation period holds {total_steps} timesteps, which is not a whole number of "
            f"{n_steps_per_cycle}-step cycles: the trailing {leftover} timestep(s) after "
            f"{last_used} are not assimilated. Adjust simulation_period or da.cycle_length to use them."
        )

    step = pd.Timedelta(seconds=dt_seconds)
    cycle_span = pd.Timedelta(seconds=cycle_seconds)
    starts = [start + i * cycle_span for i in range(n_cycles)]
    ends = [s + cycle_span - step for s in starts]

    logger.info(
        f"Cycle schedule: {n_cycles} cycles of {n_steps_per_cycle} timesteps "
        f"({cycle_length}) covering [{starts[0]}, {ends[-1]}]"
    )
    return CycleSchedule(starts=starts, ends=ends, n_steps_per_cycle=n_steps_per_cycle, dt_seconds=dt_seconds)


def slot_name(group_name: str, slot: int) -> str:
    """PEST observation name of a within-cycle slot."""
    return f"{group_name}_{slot:04d}"


def build_observation_cycle_tables(
    obs_groups: list[ObservationGroup],
    obs_frames: dict[str, pd.DataFrame],
    schedule: CycleSchedule,
    assimilate: str = "all",
    forecast_cycles: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build the per-cycle observation value and weight tables.

    Observation names are *within-cycle* slots (``{group}_{slot:04d}``, slot in
    ``0 .. n_steps_per_cycle-1``); the absolute time a slot refers to changes
    from cycle to cycle and is supplied by these tables.

    Args:
        obs_groups: Observation group configurations (for names and weights).
        obs_frames: Mapping of group name to a DataFrame with ``time`` and
            ``value`` columns (as returned by ``load_observations``).
        schedule: Cycle schedule.
        assimilate: ``"all"`` weights every slot, ``"end"`` only the last slot
            of each cycle.
        forecast_cycles: Number of trailing cycles whose weights are zeroed.

    Returns:
        Tuple ``(obs_cycle_table, weight_cycle_table, n_slots_per_group)``.
        Both tables have one row per observation name and one column per cycle;
        a slot with no matching observation gets ``NaN`` (written as an empty
        cell) in the value table and ``0.0`` in the weight table.
    """
    if assimilate not in ("all", "end"):
        raise ValueError(f"assimilate must be 'all' or 'end', got '{assimilate}'")
    if forecast_cycles > schedule.n_cycles:
        raise ValueError(f"forecast_cycles ({forecast_cycles}) exceeds the number of cycles ({schedule.n_cycles})")

    tolerance = pd.Timedelta(seconds=schedule.dt_seconds / 2)
    n_slots = schedule.n_steps_per_cycle
    last_assimilated_cycle = schedule.n_cycles - forecast_cycles

    names: list[str] = []
    values: list[list[float]] = []
    weights: list[list[float]] = []
    n_slots_per_group: dict[str, int] = {}
    matched_per_group: dict[str, int] = {}

    for group in obs_groups:
        n_slots_per_group[group.name] = n_slots
        matched_per_group[group.name] = 0

        frame = obs_frames.get(group.name)
        if frame is None or frame.empty:
            lookup = pd.Series(dtype=float)
        else:
            lookup = pd.Series(
                np.asarray(frame["value"], dtype=float),
                index=pd.DatetimeIndex(frame["time"]),
            ).sort_index()

        for slot in range(n_slots):
            names.append(slot_name(group.name, slot))
            row_values: list[float] = []
            row_weights: list[float] = []
            for cycle in range(schedule.n_cycles):
                target = schedule.starts[cycle] + pd.Timedelta(seconds=slot * schedule.dt_seconds)
                value = _nearest_value(lookup, target, tolerance)

                active = value is not None
                if active and assimilate == "end" and slot != n_slots - 1:
                    active = False
                if active and cycle >= last_assimilated_cycle:
                    active = False

                if value is not None:
                    matched_per_group[group.name] += 1
                row_values.append(np.nan if value is None else value)
                # The weight encodes the assumed observation error, which may
                # depend on the observed value itself (see observation_weights).
                row_weights.append(float(observation_weights([value], group)[0]) if active else 0.0)
            values.append(row_values)
            weights.append(row_weights)

    columns = list(range(schedule.n_cycles))
    obs_table = pd.DataFrame(values, index=names, columns=columns, dtype=float)
    weight_table = pd.DataFrame(weights, index=names, columns=columns, dtype=float)
    obs_table.index.name = "obsnme"
    weight_table.index.name = "obsnme"

    for group in obs_groups:
        logger.info(
            f"Observation group '{group.name}': {n_slots} slot(s) per cycle, "
            f"{matched_per_group[group.name]} matched observation(s) across {schedule.n_cycles} cycles"
        )
    if forecast_cycles:
        logger.info(f"Last {forecast_cycles} cycle(s) are pure forecast cycles (all weights zero)")

    return obs_table, weight_table, n_slots_per_group


def _nearest_value(lookup: pd.Series, target: pd.Timestamp, tolerance: pd.Timedelta) -> float | None:
    """Return the observed value nearest to ``target``, or None if none is within ``tolerance``."""
    if lookup.empty:
        return None
    pos = lookup.index.get_indexer([target], method="nearest")[0]
    if pos < 0:
        return None
    if abs(lookup.index[pos] - target) > tolerance:
        return None
    return float(lookup.iloc[pos])


def build_parameter_cycle_table(schedule: CycleSchedule) -> pd.DataFrame:
    """Build the per-cycle value table of the fixed ``cycle_num`` parameter.

    Args:
        schedule: Cycle schedule.

    Returns:
        One-row DataFrame indexed by ``cycle_num`` with one column per cycle.
    """
    table = pd.DataFrame(
        [[float(c) for c in range(schedule.n_cycles)]],
        index=[CYCLE_PARAM_NAME],
        columns=list(range(schedule.n_cycles)),
        dtype=float,
    )
    table.index.name = "parnme"
    return table


def assert_cycle_values_differ(par_table: pd.DataFrame) -> None:
    """Assert that no parameter repeats its value between consecutive cycles.

    PESTPP-DA may skip running the model and reuse the previous cycle's outputs
    when no dynamic states are declared and every parameter and observation
    value repeats. A distinct ``cycle_num`` per cycle guarantees this never
    happens; this check makes that guarantee explicit.

    Raises:
        ValueError: If any row repeats its value between consecutive cycles.
    """
    values = par_table.loc[CYCLE_PARAM_NAME].to_numpy(dtype=float)
    if values.size > 1 and np.any(np.diff(values) == 0):
        raise ValueError(
            f"'{CYCLE_PARAM_NAME}' repeats between consecutive cycles; PESTPP-DA would then be "
            "free to reuse the previous cycle's outputs instead of running the model"
        )


def write_cycle_tables(
    directory: Path,
    obs_table: pd.DataFrame,
    weight_table: pd.DataFrame,
    par_table: pd.DataFrame,
) -> dict[str, str]:
    """Write the three cycle tables into ``directory``.

    Args:
        directory: Working directory.
        obs_table: Per-cycle observation values.
        weight_table: Per-cycle observation weights.
        par_table: Per-cycle values of fixed parameters.

    Returns:
        Mapping of PESTPP-DA option name to the written file's basename.
    """
    directory = Path(directory)
    names = {
        "da_observation_cycle_table": "obs_cycle_table.csv",
        "da_weight_cycle_table": "weight_cycle_table.csv",
        "da_parameter_cycle_table": "par_cycle_table.csv",
    }
    for table, option in (
        (obs_table, "da_observation_cycle_table"),
        (weight_table, "da_weight_cycle_table"),
        (par_table, "da_parameter_cycle_table"),
    ):
        path = directory / names[option]
        table.to_csv(path)
        logger.info(f"Wrote {option}: {path} ({len(table)} row(s) x {len(table.columns)} cycle(s))")
    return names


def write_cycle_metadata(schedule: CycleSchedule, path: Path) -> Path:
    """Write ``da_cycles.csv``, the static cycle-to-time map read by the forward model.

    Args:
        schedule: Cycle schedule.
        path: Output CSV path.

    Returns:
        The written path.
    """
    df = pd.DataFrame(
        {
            "cycle": list(range(schedule.n_cycles)),
            "start_date": [s.isoformat(sep=" ") for s in schedule.starts],
            "end_date": [e.isoformat(sep=" ") for e in schedule.ends],
            "n_steps": [schedule.n_steps_per_cycle] * schedule.n_cycles,
        }
    )
    path = Path(path)
    df.to_csv(path, index=False)
    logger.info(f"Wrote cycle metadata: {path} ({schedule.n_cycles} cycles)")
    return path


def read_cycle_metadata(path: Path) -> pd.DataFrame:
    """Read ``da_cycles.csv`` written by :func:`write_cycle_metadata`.

    Args:
        path: Path to the CSV.

    Returns:
        DataFrame indexed by cycle with ``start_date`` / ``end_date`` as
        Timestamps and an integer ``n_steps`` column.
    """
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df.set_index("cycle")


def schedule_from_metadata(df: pd.DataFrame) -> CycleSchedule:
    """Rebuild a :class:`CycleSchedule` from the contents of ``da_cycles.csv``."""
    starts = [pd.Timestamp(t) for t in df["start_date"]]
    ends = [pd.Timestamp(t) for t in df["end_date"]]
    n_steps = int(df["n_steps"].iloc[0])
    if n_steps > 1:
        dt_seconds = int((ends[0] - starts[0]).total_seconds() / (n_steps - 1))
    elif len(starts) > 1:
        dt_seconds = int((starts[1] - starts[0]).total_seconds())
    else:
        raise ValueError("Cannot infer the timestep from a single one-step cycle")
    return CycleSchedule(starts=starts, ends=ends, n_steps_per_cycle=n_steps, dt_seconds=dt_seconds)
