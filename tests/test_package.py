"""Package-level tests for ``mobidic.__init__``.

These verify the public API surface and the optional-dependency fallback path
used when the ``pyemu`` / calibration module cannot be imported.
"""

import subprocess
import sys
import textwrap

import mobidic


class TestPublicAPI:
    """The names exported via ``__all__`` must be importable and resolvable."""

    def test_version_string(self):
        assert isinstance(mobidic.__version__, str)
        assert mobidic.__version__  # non-empty

    def test_all_names_resolve(self):
        """Every symbol in ``__all__`` must be an attribute of the package."""
        missing = [name for name in mobidic.__all__ if not hasattr(mobidic, name)]
        assert missing == [], f"Names in __all__ not bound on package: {missing}"

    def test_core_exports_present(self):
        """A few representative core/IO/preprocessing exports are reachable."""
        for name in (
            "MOBIDICConfig",
            "load_config",
            "Simulation",
            "SimulationState",
            "StateWriter",
            "load_state",
            "MeteoData",
            "MeteoRaster",
            "soil_mass_balance",
            "compute_energy_balance_1l",
        ):
            assert hasattr(mobidic, name), f"Missing public export: {name}"


class TestCalibrationImportFallback:
    """If ``pyemu`` is unavailable the package still imports and exposes its core API.

    We run the import in a subprocess with ``pyemu`` forced to fail, so the change
    cannot leak into the rest of the test session.
    """

    def test_import_succeeds_without_pyemu(self):
        script = textwrap.dedent(
            """
            import sys

            # Force ``from mobidic.calibration import ...`` to raise ImportError by
            # setting its entry in sys.modules to None before any mobidic import.
            sys.modules["mobidic.calibration"] = None

            import mobidic

            assert "CalibrationConfig" not in mobidic.__all__
            assert not hasattr(mobidic, "CalibrationConfig")
            # Core exports are still available.
            assert hasattr(mobidic, "load_config")
            assert hasattr(mobidic, "Simulation")
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert "OK" in result.stdout

    def test_import_fallback_in_process(self):
        """Re-import ``mobidic`` in-process with the calibration submodule poisoned.

        This variant (in addition to the subprocess test above) exists so that the
        ``except ImportError`` branch at the bottom of ``mobidic/__init__.py`` is
        counted by the in-process coverage tracer.
        """
        import importlib

        # Snapshot any mobidic modules that are already loaded so we can restore them.
        snapshot = {name: mod for name, mod in sys.modules.items() if name == "mobidic" or name.startswith("mobidic.")}
        try:
            # Wipe mobidic from the module cache and poison mobidic.calibration.
            for name in list(sys.modules):
                if name == "mobidic" or name.startswith("mobidic."):
                    del sys.modules[name]
            sys.modules["mobidic.calibration"] = None  # triggers ImportError

            reloaded = importlib.import_module("mobidic")
            assert "CalibrationConfig" not in reloaded.__all__
            assert not hasattr(reloaded, "CalibrationConfig")
            assert hasattr(reloaded, "load_config")
        finally:
            # Restore the original module cache to avoid leaking state to other tests.
            for name in list(sys.modules):
                if name == "mobidic" or name.startswith("mobidic."):
                    del sys.modules[name]
            sys.modules.update(snapshot)
