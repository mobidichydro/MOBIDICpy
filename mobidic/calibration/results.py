"""CalibrationResults: parse and access PEST++ output files."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mobidic.calibration.config import CalibrationConfig


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

            pst_path = self.master_dir / "calibration.pst"
            if pst_path.exists():
                self._pst = pyemu.Pst(str(pst_path))
            else:
                logger.warning(f"PST file not found: {pst_path}")
        return self._pst

    def get_optimal_parameters(self) -> dict[str, float]:
        """Get the optimal parameter values from the calibration.

        Returns:
            Dict mapping parameter name to optimal value.
        """
        # Try to read from .par file (final parameter values)
        par_files = sorted(self.master_dir.glob("calibration.*.par"))
        if not par_files:
            par_files = sorted(self.master_dir.glob("*.par"))

        if par_files:
            # Use the last .par file
            par_file = par_files[-1]
            logger.info(f"Reading optimal parameters from: {par_file}")
            return self._parse_par_file(par_file)

        # Fallback: read from .pst parameter_data (initial values)
        if self.pst is not None:
            return dict(zip(self.pst.parameter_data.index, self.pst.parameter_data["parval1"]))

        return {}

    def get_objective_function_history(self) -> pd.DataFrame | None:
        """Get objective function values across iterations.

        Returns:
            DataFrame with iteration number and phi (objective function value),
            or None if record file not found.
        """
        rec_path = self.master_dir / "calibration.rec"
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
        rei_files = sorted(self.master_dir.glob("calibration.*.rei"))
        if not rei_files:
            rei_files = sorted(self.master_dir.glob("*.rei"))

        if not rei_files:
            logger.warning("No .rei (residuals) files found")
            return None

        rei_file = rei_files[-1]
        logger.info(f"Reading residuals from: {rei_file}")
        return self._parse_rei_file(rei_file)

    def get_parameter_sensitivities(self) -> pd.DataFrame | None:
        """Get parameter sensitivities (from pestpp-sen or GLM Jacobian).

        Returns:
            DataFrame with sensitivity information, or None if not available.
        """
        # pestpp-sen output
        sen_file = self.master_dir / "calibration.sn"
        if sen_file.exists():
            return pd.read_csv(sen_file)

        # GLM: try to load Jacobian via pyemu
        jco_path = self.master_dir / "calibration.jcb"
        if not jco_path.exists():
            jco_path = self.master_dir / "calibration.jco"

        if jco_path.exists():
            import pyemu

            jco = pyemu.Jco.from_binary(str(jco_path))
            # Composite sensitivity: column-wise L2 norm
            sens = pd.DataFrame(
                {
                    "parameter": jco.col_names,
                    "sensitivity": np.sqrt(np.asarray((jco.x**2).sum(axis=0)).ravel()),
                }
            )
            return sens.sort_values("sensitivity", ascending=False).reset_index(drop=True)

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
        prior_par = self.master_dir / "calibration.0.par.csv"
        if prior_par.exists():
            results["prior_parameters"] = pd.read_csv(prior_par, index_col=0)

        # Find last iteration's parameter ensemble
        par_csvs = sorted(self.master_dir.glob("calibration.*.par.csv"))
        if par_csvs:
            last_par = par_csvs[-1]
            results["posterior_parameters"] = pd.read_csv(last_par, index_col=0)

        # Observation ensembles
        prior_obs = self.master_dir / "calibration.0.obs.csv"
        if prior_obs.exists():
            results["prior_observations"] = pd.read_csv(prior_obs, index_col=0)

        obs_csvs = sorted(self.master_dir.glob("calibration.*.obs.csv"))
        if obs_csvs:
            last_obs = obs_csvs[-1]
            results["posterior_observations"] = pd.read_csv(last_obs, index_col=0)

        if not results:
            return None
        return results

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
