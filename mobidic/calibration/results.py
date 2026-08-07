"""CalibrationResults: parse and access PEST++ output files."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mobidic.calibration.config import CalibrationConfig


def _is_da_parameter(name: str) -> bool:
    """True for PEST parameters belonging to the data-assimilation interface."""
    from mobidic.calibration.da_cycles import CYCLE_PARAM_NAME
    from mobidic.calibration.da_states import STATE_PAR_PREFIX

    lowered = name.lower()
    return lowered == CYCLE_PARAM_NAME or lowered.startswith(STATE_PAR_PREFIX)


def _parse_cycle_iteration(filename: str, case: str, suffix: str) -> tuple[int, int] | None:
    """Parse ``{case}.{cycle}.{iteration}.{suffix}`` into ``(cycle, iteration)``.

    With ``noptmax = 0`` PESTPP-DA performs a single run per cycle at the
    control-file parameter values and names the file after the realization
    instead of an iteration (``{case}.{cycle}.base.obs.csv``). That form is
    reported as iteration 0; the two never coexist, since noptmax = 0 produces
    no numbered iterations.
    """
    prefix = f"{case}."
    if not filename.startswith(prefix) or not filename.endswith(f".{suffix}"):
        return None
    middle = filename[len(prefix) : -len(suffix) - 1]
    parts = middle.split(".")
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    cycle = int(parts[0])
    if parts[1].isdigit():
        return cycle, int(parts[1])
    return cycle, 0


def _slot_index(obs_name: str, group: str) -> int | None:
    """Return the within-cycle slot index of ``obs_name`` in ``group``, or None."""
    prefix = f"{group.lower()}_"
    lowered = obs_name.lower()
    if not lowered.startswith(prefix):
        return None
    tail = lowered[len(prefix) :]
    return int(tail) if tail.isdigit() else None


class CalibrationResults:
    """Container for parsed PEST++ calibration results.

    Provides access to:
    - Optimal parameter values
    - Objective function history
    - Residuals
    - Parameter sensitivities (GLM/SEN)
    - Ensemble statistics (IES)

    Args:
        master_dir: Path to the PEST++ master directory.
        calib_config: Calibration configuration.
    """

    def __init__(self, master_dir: Path, calib_config: CalibrationConfig):
        self.master_dir = Path(master_dir)
        self.calib_config = calib_config
        self._pst = None
        self._rec_data = None

    @classmethod
    def from_pest_output(cls, master_dir: Path, calib_config: CalibrationConfig) -> "CalibrationResults":
        """Create CalibrationResults from completed PEST++ output.

        Args:
            master_dir: Path to PEST++ master directory.
            calib_config: Calibration configuration.

        Returns:
            CalibrationResults object.
        """
        return cls(master_dir=master_dir, calib_config=calib_config)

    @property
    def pst(self):
        """Load the PEST control file."""
        if self._pst is None:
            import pyemu

            pst_path = self.master_dir / f"{self.calib_config.case_name}.pst"
            if pst_path.exists():
                self._pst = pyemu.Pst(str(pst_path))
            else:
                logger.warning(f"PST file not found: {pst_path}")
        return self._pst

    def get_optimal_parameters(self) -> dict[str, float]:
        """Get the optimal parameter values from the calibration.

        PEST lowercases all parameter names, so the mapping from PEST
        names back to ``parameter_key`` is done case-insensitively.

        Returns:
            Dict mapping each parameter's ``parameter_key`` (dot-notation
            path into the MOBIDIC YAML config) to its optimal value.

        Raises:
            KeyError: If a PEST parameter has no matching entry in
                ``calib_config.parameters``.
        """
        if self.calib_config.pest_tool == "swp":
            logger.info("pest_tool='swp' performs no optimization; use get_sweep_results().")
            return {}

        if self.calib_config.is_sequential_da:
            logger.info(
                "Sequential data assimilation has no single optimum: every cycle has its own "
                "posterior. Use get_da_results() / get_da_timeseries()."
            )
            return {}

        # Try to read from .par file (final parameter values)
        par_files = sorted(self.master_dir.glob(f"{self.calib_config.case_name}.*.par"))
        if not par_files:
            par_files = sorted(self.master_dir.glob("*.par"))

        if par_files:
            # Use the last .par file
            par_file = par_files[-1]
            logger.info(f"Reading optimal parameters from: {par_file}")
            raw = self._parse_par_file(par_file)
        elif self.pst is not None:
            # Fallback: read from .pst parameter_data (initial values)
            raw = dict(zip(self.pst.parameter_data.index, self.pst.parameter_data["parval1"]))
        else:
            return {}

        name_to_key = {p.name.lower(): p.parameter_key for p in self.calib_config.parameters}
        result = {}
        for name, value in raw.items():
            if _is_da_parameter(name):
                # cycle_num and the state parameters are part of the data
                # assimilation interface, not of the MOBIDIC configuration.
                continue
            key = name_to_key.get(name.lower())
            if key is None:
                raise KeyError(
                    f"PEST parameter '{name}' has no matching entry in calib_config.parameters. "
                    f"Known parameters: {sorted(name_to_key)}"
                )
            result[key] = value
        return result

    def get_da_results(self, cycle: int | None = None, iteration: int | None = None) -> dict:
        """Read the per-cycle PESTPP-DA ensembles.

        PESTPP-DA writes ``{case}.{cycle}.{iteration}.par.csv`` / ``.obs.csv``
        for every cycle and iteration, plus ``{case}.global.{cycle}.pe.csv`` /
        ``.oe.csv`` holding the ensembles at the end of each cycle.

        Args:
            cycle: Restrict to a single cycle (default: all cycles found).
            iteration: Restrict to a single iteration (default: all).

        Returns:
            Dict with keys ``"parameters"`` and ``"observations"``, each mapping
            ``(cycle, iteration)`` to a DataFrame (realizations x names), plus
            ``"global_parameters"`` and ``"global_observations"`` mapping cycle
            to a DataFrame. Empty dicts when no file is found.
        """
        case = self.calib_config.case_name
        out: dict[str, dict] = {
            "parameters": {},
            "observations": {},
            "global_parameters": {},
            "global_observations": {},
        }

        for kind, suffix in (("parameters", "par"), ("observations", "obs")):
            for path in sorted(self.master_dir.glob(f"{case}.*.*.{suffix}.csv")):
                key = _parse_cycle_iteration(path.name, case, f"{suffix}.csv")
                if key is None:
                    continue
                if cycle is not None and key[0] != cycle:
                    continue
                if iteration is not None and key[1] != iteration:
                    continue
                out[kind][key] = pd.read_csv(path, index_col=0)

        for kind, suffix in (("global_parameters", "pe"), ("global_observations", "oe")):
            for path in sorted(self.master_dir.glob(f"{case}.global.*.{suffix}.csv")):
                stem = path.name[len(f"{case}.global.") : -len(f".{suffix}.csv")]
                if not stem.isdigit():
                    continue
                c = int(stem)
                if cycle is not None and c != cycle:
                    continue
                out[kind][c] = pd.read_csv(path, index_col=0)

        found = sum(len(v) for v in out.values())
        if found == 0:
            logger.warning(f"No PESTPP-DA ensemble files found in {self.master_dir} for case '{case}'")
        else:
            logger.info(f"Read {found} PESTPP-DA ensemble file(s) from {self.master_dir}")
        return out

    def get_da_timeseries(self, group: str, posterior: bool = True) -> pd.DataFrame | None:
        """Reassemble the per-cycle observation ensembles into a continuous series.

        Each cycle reports the same within-cycle observation slots
        (``{group}_{slot:04d}``); this method maps them back onto absolute time
        using ``da_cycles.csv`` and concatenates the cycles in order.

        Args:
            group: Observation group name.
            posterior: If True use the last iteration of each cycle (posterior),
                otherwise iteration 0 (prior).

        Returns:
            DataFrame indexed by absolute time with one column per realization,
            or None if the cycle metadata or the ensembles are unavailable.
        """
        from mobidic.calibration.da_cycles import CYCLE_METADATA_FILE, read_cycle_metadata

        cycles_path = self.master_dir / CYCLE_METADATA_FILE
        if not cycles_path.exists():
            logger.warning(f"Cycle metadata not found: {cycles_path}")
            return None
        cycles = read_cycle_metadata(cycles_path)

        ensembles = self.get_da_results()["observations"]
        if not ensembles:
            return None

        by_cycle: dict[int, int] = {}
        for c, it in ensembles:
            if c not in by_cycle:
                by_cycle[c] = it
            elif posterior:
                by_cycle[c] = max(by_cycle[c], it)
            else:
                by_cycle[c] = min(by_cycle[c], it)

        frames = []
        for c in sorted(by_cycle):
            if c not in cycles.index:
                continue
            df = ensembles[(c, by_cycle[c])]
            columns = [name for name in df.columns if _slot_index(name, group) is not None]
            if not columns:
                continue
            columns.sort(key=lambda name: _slot_index(name, group))
            n_steps = int(cycles.loc[c, "n_steps"])
            start = cycles.loc[c, "start_date"]
            end = cycles.loc[c, "end_date"]
            dt = (end - start) / max(n_steps - 1, 1)
            times = pd.date_range(start=start, periods=len(columns), freq=dt)
            block = df[columns].T
            block.index = times
            frames.append(block)

        if not frames:
            logger.warning(f"No observation slots found for group '{group}'")
            return None

        result = pd.concat(frames).sort_index()
        result.index.name = "time"
        logger.info(
            f"Reassembled {'posterior' if posterior else 'prior'} time series for '{group}': "
            f"{len(result)} timesteps x {result.shape[1]} realizations"
        )
        return result

    def get_da_state_ids(self, cycle: int, iteration: int | None = None) -> pd.Series | None:
        """Get the state-file identifier each realization holds at the end of a cycle.

        Only meaningful with ``da.states.restart_from='previous_cycle'``, where
        the full state is carried in ``da_states/c{cycle:04d}_{id:06d}.npz`` and
        only this identifier travels through the PEST interface. Use it to
        launch a forecast from the analysed state of a given cycle.

        Args:
            cycle: Cycle whose *final* state is wanted.
            iteration: Iteration to read (default: the last one, i.e. the
                posterior).

        Returns:
            Series mapping realization name to integer state identifier, or
            None if the cycle's observation ensemble is unavailable.
        """
        from mobidic.calibration.da_states import STATE_ID_OBS_NAME

        ensembles = self.get_da_results(cycle=cycle)["observations"]
        if not ensembles:
            logger.warning(f"No observation ensemble found for cycle {cycle}")
            return None

        if iteration is None:
            iteration = max(it for _, it in ensembles)
        df = ensembles[(cycle, iteration)]

        column = next((c for c in df.columns if c.lower() == STATE_ID_OBS_NAME), None)
        if column is None:
            logger.warning(
                f"'{STATE_ID_OBS_NAME}' is not in the cycle {cycle} ensemble; "
                "state identifiers only exist with restart_from='previous_cycle'"
            )
            return None

        return df[column].round().astype(int)

    def get_da_states(
        self,
        cycle: int,
        kind: str = "discharge",
        iteration: int | None = None,
        simulated: bool = False,
    ) -> pd.DataFrame | None:
        """Get the estimated state ensemble of a cycle.

        Only meaningful with joint state-parameter estimation
        (``da.states.estimate``), where each estimated state is an adjustable
        PEST parameter. Two views of the same cycle are available:

        - ``simulated=False`` (default): the *analysed initial* state, read from
          the parameter ensemble. This is what the filter produced and what the
          cycle was run with.
        - ``simulated=True``: the *simulated final* state, read from the
          observation ensemble. This is what PESTPP-DA transfers into the next
          cycle.

        Args:
            cycle: Cycle to read.
            kind: State variable: ``"discharge"`` [m3/s] per reach, or
                ``"soil_capillary"`` / ``"soil_gravitational"`` (zone-averaged
                saturation in ``[0, 1]``).
            iteration: Iteration to read (default: the last one, the posterior).
            simulated: Read the simulated final states instead of the analysed
                initial states.

        Returns:
            DataFrame indexed by realization with one integer-named column per
            unit (reach id for discharge, zone id for the soil stores; with
            ``zones: reach`` a zone id is the id of the reach its cells drain
            to), or None if the cycle's ensemble is unavailable.
        """
        from mobidic.calibration.da_states import STATE_KIND_TAGS, STATE_OBS_PREFIX, STATE_PAR_PREFIX

        tag = STATE_KIND_TAGS.get(kind)
        if tag is None:
            raise ValueError(f"Unknown state kind '{kind}' (supported: {sorted(STATE_KIND_TAGS)})")

        key = "observations" if simulated else "parameters"
        ensembles = self.get_da_results(cycle=cycle)[key]
        if not ensembles:
            logger.warning(f"No {key[:-1]} ensemble found for cycle {cycle}")
            return None

        if iteration is None:
            iteration = max(it for _, it in ensembles)
        df = ensembles[(cycle, iteration)]

        prefix = f"{STATE_OBS_PREFIX if simulated else STATE_PAR_PREFIX}{tag}_"
        columns = {c: int(c.lower()[len(prefix) :]) for c in df.columns if c.lower().startswith(prefix)}
        if not columns:
            logger.warning(
                f"No '{kind}' state columns in the cycle {cycle} ensemble; states are only present "
                "with da.states.estimate set"
            )
            return None

        states = df[list(columns)].rename(columns=columns)
        return states[sorted(states.columns)]

    def get_objective_function_history(self) -> pd.DataFrame | None:
        """Get objective function values across iterations.

        Returns a DataFrame with at minimum ``iteration`` and ``phi`` columns.
        The source file and meaning of ``phi`` depend on the PEST++ tool:

        - ``glm``: reads ``calibration.iobj`` (CSV); ``phi`` = ``total_phi``.
        - ``ies``: reads ``calibration.phi.actual.csv``; ``phi`` = mean across
          ensemble members. Extra columns ``std`` and one column per member are
          also included.
        - other tools: reads ``calibration.rec``; ``phi`` = total phi extracted
          from the record file.

        Returns:
            DataFrame with iteration number and phi (objective function value),
            or None if the expected file is not found.
        """
        match self.calib_config.pest_tool:
            case "glm":
                iobj_path = self.master_dir / f"{self.calib_config.case_name}.iobj"
                if not iobj_path.exists():
                    logger.warning(f"iobj file not found: {iobj_path}")
                    return None
                return self._parse_iobj_phi(iobj_path)

            case "ies":
                phi_path = self.master_dir / f"{self.calib_config.case_name}.phi.actual.csv"
                if not phi_path.exists():
                    logger.warning(f"IES phi file not found: {phi_path}")
                    return None
                return self._parse_ies_phi(phi_path)

            case "da":
                # PESTPP-DA writes the same phi format as IES, with one block
                # per cycle in {case}.global.phi.actual.csv.
                phi_path = self.master_dir / f"{self.calib_config.case_name}.global.phi.actual.csv"
                if not phi_path.exists():
                    phi_path = self.master_dir / f"{self.calib_config.case_name}.phi.actual.csv"
                if not phi_path.exists():
                    logger.warning(f"DA phi file not found in {self.master_dir}")
                    return None
                return self._parse_ies_phi(phi_path)

            case _:
                rec_path = self.master_dir / f"{self.calib_config.case_name}.rec"
                if not rec_path.exists():
                    logger.warning(f"Record file not found: {rec_path}")
                    return None
                return self._parse_rec_phi(rec_path)

    def get_residuals(self) -> pd.DataFrame | None:
        """Get observation residuals (simulated - observed) from the final iteration.

        Returns:
            DataFrame with obs_name, observed, simulated, residual, weight columns,
            or None if .rei file not found.
        """
        rei_files = sorted(self.master_dir.glob(f"{self.calib_config.case_name}.*.rei"))
        if not rei_files:
            rei_files = sorted(self.master_dir.glob("*.rei"))

        if not rei_files:
            logger.warning("No .rei (residuals) files found")
            return None

        rei_file = rei_files[-1]
        logger.info(f"Reading residuals from: {rei_file}")
        return self._parse_rei_file(rei_file)

    def get_parameter_sensitivities(self) -> pd.DataFrame | None:
        """Get parameter sensitivities.

        - ``sen``: reads ``{case}.msn`` (Morris sensitivity); returns all columns
          (par_name, n_samples, sen_mean, sen_mean_abs, sen_std_dev).
        - ``glm``: loads the Jacobian (``.jcb`` / ``.jco``) via pyemu and computes
          composite sensitivity (column-wise L2 norm).

        Returns:
            DataFrame with sensitivity information, or None if not available.
        """
        match self.calib_config.pest_tool:
            case "sen":
                msn_path = self.master_dir / f"{self.calib_config.case_name}.msn"
                if not msn_path.exists():
                    logger.warning(f"Morris sensitivity file not found: {msn_path}")
                    return None
                df = pd.read_csv(msn_path)
                logger.info(f"Read Morris sensitivities from {msn_path}: {len(df)} parameters")
                return df

            case "glm":
                jco_path = self.master_dir / f"{self.calib_config.case_name}.jcb"
                if not jco_path.exists():
                    jco_path = self.master_dir / f"{self.calib_config.case_name}.jco"
                if not jco_path.exists():
                    logger.warning(f"Jacobian file not found in {self.master_dir}")
                    return None
                import pyemu

                jco = pyemu.Jco.from_binary(str(jco_path))
                sens = pd.DataFrame(
                    {
                        "parameter": jco.col_names,
                        "sensitivity": np.sqrt(np.asarray((jco.x**2).sum(axis=0)).ravel()),
                    }
                )
                logger.info(f"Computed GLM composite sensitivities from {jco_path}: {len(sens)} parameters")
                return sens.sort_values("sensitivity", ascending=False).reset_index(drop=True)

            case _:
                logger.warning(f"Sensitivity output not supported for pest_tool='{self.calib_config.pest_tool}'")
                return None

    def get_ensemble_results(self) -> dict | None:
        """Get IES ensemble results (prior and posterior).

        Returns:
            Dict with 'prior_parameters', 'posterior_parameters',
            'prior_observations', 'posterior_observations' DataFrames,
            or None if not available.
        """

        results = {}

        # Parameter ensembles
        prior_par = self.master_dir / f"{self.calib_config.case_name}.0.par.csv"
        if prior_par.exists():
            results["prior_parameters"] = pd.read_csv(prior_par, index_col=0)

        # Find last iteration's parameter ensemble
        par_csvs = sorted(self.master_dir.glob(f"{self.calib_config.case_name}.*.par.csv"))
        if par_csvs:
            last_par = par_csvs[-1]
            results["posterior_parameters"] = pd.read_csv(last_par, index_col=0)

        # Observation ensembles
        prior_obs = self.master_dir / f"{self.calib_config.case_name}.0.obs.csv"
        if prior_obs.exists():
            results["prior_observations"] = pd.read_csv(prior_obs, index_col=0)

        obs_csvs = sorted(self.master_dir.glob(f"{self.calib_config.case_name}.*.obs.csv"))
        if obs_csvs:
            last_obs = obs_csvs[-1]
            results["posterior_observations"] = pd.read_csv(last_obs, index_col=0)

        if not results:
            return None
        return results

    def get_sweep_results(self) -> pd.DataFrame | None:
        """Read the pestpp-swp output ensemble (``{case}.sweep_out.csv``).

        Each row corresponds to one forward run (one row of the input parameter
        sweep CSV) and contains the run id, phi, and one column per observation.

        Returns:
            DataFrame of sweep outputs, or None if the file is not found.
        """
        path = self.master_dir / "sweep_out.csv"
        if not path.exists():
            logger.warning(f"Sweep output file not found: {path}")
            return None
        df = pd.read_csv(path)
        logger.info(f"Read sweep output from {path}: {len(df)} runs")
        return df

    def _parse_par_file(self, par_file: Path) -> dict[str, float]:
        """Parse a PEST .par file to extract parameter values."""
        params = {}
        with open(par_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Skip header line ("single point")
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                name = parts[0]
                value = float(parts[1])
                params[name] = value
        return params

    def _parse_ies_phi(self, phi_path: Path) -> pd.DataFrame:
        """Parse objective function history from IES calibration.phi.actual.csv."""
        df = pd.read_csv(phi_path)
        # Skip first 6 columns
        num_members = len(df.columns) - 6
        # Add phi and std columns for consistency with GLM
        df["phi"] = df["mean"]
        df["std"] = df["standard_deviation"]
        logger.info(f"Read IES phi history from {phi_path}: {len(df)} iterations, {num_members} ensemble members")
        return df

    def _parse_iobj_phi(self, iobj_path: Path) -> pd.DataFrame:
        """Parse objective function history from GLM .iobj CSV file."""
        df = pd.read_csv(iobj_path)
        result = df[["iteration", "total_phi"]].rename(columns={"total_phi": "phi"})
        logger.info(f"Read GLM phi history from {iobj_path}: {len(result)} iterations")
        return result

    def _parse_rec_phi(self, rec_path: Path) -> pd.DataFrame:
        """Parse objective function history from .rec file."""
        iterations = []
        phis = []
        with open(rec_path, "r", encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip().lower()
                if "starting phi for this iteration" in line_stripped or "total phi" in line_stripped:
                    # Try to extract the phi value
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        try:
                            phi = float(part)
                            phis.append(phi)
                            iterations.append(len(phis))
                            break
                        except ValueError:
                            continue

        return pd.DataFrame({"iteration": iterations, "phi": phis})

    def _parse_rei_file(self, rei_file: Path) -> pd.DataFrame:
        """Parse a PEST .rei residuals file."""
        rows = []
        with open(rei_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find data start (after header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Name"):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 6:
                rows.append(
                    {
                        "obs_name": parts[0],
                        "group": parts[1],
                        "observed": float(parts[2]),
                        "simulated": float(parts[3]),
                        "residual": float(parts[4]),
                        "weight": float(parts[5]),
                    }
                )

        return pd.DataFrame(rows) if rows else None
