"""Tests for the Numba JIT enable/disable switch."""

import numpy as np
import pytest

from mobidic.core.routing import _hillslope_routing_kernel, _linear_routing_kernel, hillslope_routing
from mobidic.utils import jit as jit_module
from mobidic.utils.jit import dispatch, jit_enabled, set_jit_enabled


@pytest.fixture(autouse=True)
def restore_jit_state(monkeypatch):
    """Restore the global JIT flag (and environment) after each test."""
    monkeypatch.delenv("NUMBA_DISABLE_JIT", raising=False)
    original = jit_module._JIT_ENABLED
    yield
    jit_module._JIT_ENABLED = original


class TestJitFlag:
    """Tests for the module-level enable/disable flag."""

    def test_enabled_by_default(self):
        """JIT is enabled when NUMBA_DISABLE_JIT is not set."""
        assert jit_enabled() is True

    def test_set_disabled(self):
        """Disabling the flag is reflected by jit_enabled()."""
        set_jit_enabled(False)
        assert jit_enabled() is False

    def test_re_enable(self):
        """The flag can be toggled back on."""
        set_jit_enabled(False)
        set_jit_enabled(True)
        assert jit_enabled() is True

    def test_env_var_overrides_config(self, monkeypatch):
        """NUMBA_DISABLE_JIT=1 wins over an explicit enable request."""
        monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
        set_jit_enabled(True)
        assert jit_enabled() is False

    def test_env_var_zero_is_not_disabling(self, monkeypatch):
        """NUMBA_DISABLE_JIT=0 leaves JIT enabled."""
        monkeypatch.setenv("NUMBA_DISABLE_JIT", "0")
        set_jit_enabled(True)
        assert jit_enabled() is True


class TestDispatch:
    """Tests for kernel dispatching."""

    def test_returns_dispatcher_when_enabled(self):
        """With JIT on, the compiled dispatcher itself is returned."""
        set_jit_enabled(True)
        assert dispatch(_linear_routing_kernel) is _linear_routing_kernel

    def test_returns_py_func_when_disabled(self):
        """With JIT off, the underlying Python function is returned."""
        set_jit_enabled(False)
        assert dispatch(_linear_routing_kernel) is _linear_routing_kernel.py_func

    def test_plain_function_without_py_func(self):
        """A plain function (as produced under NUMBA_DISABLE_JIT) is returned unchanged."""
        set_jit_enabled(False)

        def plain(x):
            return x

        assert dispatch(plain) is plain

    def test_disabled_dispatch_does_not_compile(self):
        """Selecting py_func must not trigger Numba compilation."""
        set_jit_enabled(False)
        n_signatures = len(_hillslope_routing_kernel.signatures)

        lateral_flow = np.ones((3, 3))
        flow_direction = np.array([[5.0, 6.0, 7.0], [4.0, 0.0, 8.0], [3.0, 2.0, 1.0]])
        upstream = np.zeros((3, 3))
        dispatch(_hillslope_routing_kernel)(lateral_flow, flow_direction, upstream, 3, 3)

        assert len(_hillslope_routing_kernel.signatures) == n_signatures


class TestKernelEquivalence:
    """Both modes must produce the same numerical results."""

    def test_hillslope_routing_matches(self):
        """Hillslope routing gives identical results with and without JIT."""
        rng = np.random.default_rng(42)
        lateral_flow = rng.random((12, 15))
        flow_direction = rng.integers(1, 9, size=(12, 15)).astype(float)
        flow_direction[5, 5] = 0.0
        flow_direction[0, 0] = np.nan

        set_jit_enabled(True)
        expected = hillslope_routing(lateral_flow, flow_direction)

        set_jit_enabled(False)
        actual = hillslope_routing(lateral_flow, flow_direction)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    def test_linear_routing_matches(self):
        """Linear channel routing gives identical results with and without JIT."""
        from mobidic.core.routing import linear_channel_routing

        # Chain of three reaches: 0 -> 1 -> 2
        network = {
            "n_reaches": 3,
            "upstream_1_idx": np.array([-1, 0, 1], dtype=np.int32),
            "upstream_2_idx": np.array([-1, -1, -1], dtype=np.int32),
            "n_upstream": np.array([0, 1, 1], dtype=np.int32),
            "sorted_reach_idx": np.array([0, 1, 2], dtype=np.int32),
            "K": np.array([3600.0, 7200.0, 5400.0]),
        }
        discharge_initial = np.array([10.0, 5.0, 2.0])
        lateral_inflow = np.array([2.0, 1.0, 0.5])

        set_jit_enabled(True)
        expected, _ = linear_channel_routing(network, discharge_initial, lateral_inflow, dt=900.0)

        set_jit_enabled(False)
        actual, _ = linear_channel_routing(network, discharge_initial, lateral_inflow, dt=900.0)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)

    def test_energy_balance_matches(self):
        """The 1L energy balance kernel agrees between modes (fastmath tolerance)."""
        from mobidic.core.energy_balance import energy_balance_1l

        n = 8
        rng = np.random.default_rng(7)
        kwargs = dict(
            ff=2 * np.pi / 86400.0,
            a_tem=np.full(n, 5.0),
            a_rad=np.full(n, 300.0),
            p_tem=0.5,
            p_rad=1.0,
            c_tem=np.full(n, 0.5),
            c_rad=np.full(n, 0.5),
            td_ini=np.full(n, 288.0),
            tm=288.0 + rng.random(n),
            u=np.full(n, 2.0),
            pair=1013.0,
            hair=np.full(n, 0.6),
            step=600.0,
            ch=np.full(n, 1e-3),
            alb=np.full(n, 0.2),
            kaps=2.5,
            nis=0.8e-6,
            tcost=290.0,
            etrsuetp=np.full(n, 0.5),
            t_end=3600.0,
        )

        set_jit_enabled(True)
        ts_jit, td_jit, evp_jit = energy_balance_1l(**kwargs)

        set_jit_enabled(False)
        ts_py, td_py, evp_py = energy_balance_1l(**kwargs)

        # fastmath/parallel reorder the reductions, so results agree to float tolerance
        np.testing.assert_allclose(ts_py, ts_jit, rtol=1e-10)
        np.testing.assert_allclose(td_py, td_jit, rtol=1e-10)
        np.testing.assert_allclose(evp_py, evp_jit, rtol=1e-10)
