"""Tests for PESTPP-DA state files."""

from types import SimpleNamespace

import numpy as np
import pytest

from mobidic.calibration.da_states import (
    MAX_STATE_ID,
    build_state_mask,
    new_state_id,
    read_state_file,
    remove_old_state_files,
    state_file_path,
    write_state_file,
)
from mobidic.core.simulation import SimulationState

SHAPE = (4, 4)


def _mask():
    """Active in the top three rows; the bottom row is outside the domain."""
    mask = np.ones(SHAPE, dtype=bool)
    mask[3, :] = False
    return mask


def _grid(seed, mask, fill=np.nan):
    rng = np.random.default_rng(seed)
    grid = np.full(SHAPE, fill, dtype=np.float64)
    grid[mask] = rng.random(int(mask.sum()))
    return grid


def _state(mask, n_reaches=5, reservoirs=False):
    from mobidic.core.reservoir import ReservoirState

    reservoir_states = None
    if reservoirs:
        reservoir_states = [ReservoirState(volume=1.5e6, stage=231.4, inflow=2.0, outflow=1.0)]

    return SimulationState(
        wc=_grid(1, mask),
        wg=_grid(2, mask),
        wp=_grid(3, mask),
        ws=_grid(4, mask),
        discharge=np.arange(n_reaches, dtype=np.float64),
        lateral_inflow=np.arange(n_reaches, dtype=np.float64) * 0.5,
        reservoir_states=reservoir_states,
        flr=_grid(5, mask),
        fld=_grid(6, mask),
    )


class TestStateFileRoundTrip:
    def test_roundtrip_is_exact_including_the_nan_mask(self, tmp_path):
        mask = _mask()
        state = _state(mask)
        path = write_state_file(tmp_path / "s.npz", state, mask)

        loaded = read_state_file(path, mask)

        for name in ("wc", "wg", "wp", "ws", "flr", "fld"):
            original = getattr(state, name)
            restored = getattr(loaded, name)
            np.testing.assert_array_equal(np.isnan(original), np.isnan(restored))
            np.testing.assert_array_equal(restored, original)
            # The row outside the domain is NaN in both
            assert np.all(np.isnan(restored[3, :]))
        np.testing.assert_array_equal(loaded.discharge, state.discharge)
        np.testing.assert_array_equal(loaded.lateral_inflow, state.lateral_inflow)

    def test_roundtrip_survives_a_non_nan_outside_fill(self, tmp_path):
        mask = _mask()
        state = _state(mask)
        state.flr = _grid(7, mask, fill=0.0)
        path = write_state_file(tmp_path / "s.npz", state, mask)

        loaded = read_state_file(path, mask)
        np.testing.assert_array_equal(loaded.flr, state.flr)
        assert np.all(loaded.flr[3, :] == 0.0)

    def test_roundtrip_survives_a_non_uniform_outside_region(self, tmp_path):
        mask = _mask()
        state = _state(mask)
        state.ws = state.ws.copy()
        state.ws[3, 0] = 7.0
        state.ws[3, 1] = -3.0
        path = write_state_file(tmp_path / "s.npz", state, mask)

        loaded = read_state_file(path, mask)
        np.testing.assert_array_equal(loaded.ws, state.ws)

    def test_reservoir_states_roundtrip(self, tmp_path):
        mask = _mask()
        state = _state(mask, reservoirs=True)
        path = write_state_file(tmp_path / "s.npz", state, mask)

        loaded = read_state_file(path, mask)
        assert len(loaded.reservoir_states) == 1
        assert loaded.reservoir_states[0].volume == pytest.approx(1.5e6)
        assert loaded.reservoir_states[0].stage == pytest.approx(231.4)
        assert loaded.reservoir_states[0].outflow == pytest.approx(1.0)

    def test_optional_variables_absent_from_the_file_come_from_the_template(self, tmp_path):
        mask = _mask()
        state = _state(mask)
        state.wp = None
        path = write_state_file(tmp_path / "s.npz", state, mask)

        template = _state(mask)
        loaded = read_state_file(path, mask, template=template)
        np.testing.assert_array_equal(loaded.wp, template.wp)

    def test_missing_flr_is_rejected_at_write_time(self, tmp_path):
        mask = _mask()
        state = _state(mask)
        state.flr = None
        with pytest.raises(ValueError, match="required state variable"):
            write_state_file(tmp_path / "s.npz", state, mask)

    def test_missing_file_raises_instead_of_falling_back(self, tmp_path):
        mask = _mask()
        with pytest.raises(FileNotFoundError, match="State file not found"):
            read_state_file(tmp_path / "absent.npz", mask)

    def test_shape_mismatch_is_detected(self, tmp_path):
        mask = _mask()
        path = write_state_file(tmp_path / "s.npz", _state(mask), mask)
        with pytest.raises(ValueError, match="grid shape"):
            read_state_file(path, np.ones((5, 5), dtype=bool))

    def test_active_cell_count_mismatch_is_detected(self, tmp_path):
        """A different mask of the same shape must fail loudly, not scatter wrongly."""
        mask = _mask()
        path = write_state_file(tmp_path / "s.npz", _state(mask), mask)

        wider = mask.copy()
        wider[3, 0] = True  # one more active cell, same grid shape

        with pytest.raises(ValueError, match="active grid cell"):
            read_state_file(path, wider)


class TestStateIdentifiers:
    def test_new_state_id_is_in_range_and_creates_the_file(self, tmp_path):
        sid = new_state_id(tmp_path, cycle=3)
        assert 0 <= sid < MAX_STATE_ID
        assert state_file_path(tmp_path, 3, sid).exists()

    def test_new_state_id_never_reuses_a_claimed_number(self, tmp_path):
        ids = {new_state_id(tmp_path, cycle=0) for _ in range(50)}
        assert len(ids) == 50

    def test_state_file_path_is_zero_padded(self, tmp_path):
        assert state_file_path(tmp_path, 7, 42).name == "c0007_000042.npz"


class TestRemoveOldStateFiles:
    def test_keeps_exactly_keep_cycles(self, tmp_path):
        for cycle in range(6):
            state_file_path(tmp_path, cycle, 1).write_bytes(b"")

        removed = remove_old_state_files(tmp_path, cycle=5, keep=2)

        remaining = sorted(p.name for p in tmp_path.glob("c*.npz"))
        assert remaining == ["c0004_000001.npz", "c0005_000001.npz"]
        assert removed == 4

    def test_ignores_unrelated_files(self, tmp_path):
        state_file_path(tmp_path, 0, 1).write_bytes(b"")
        (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")

        remove_old_state_files(tmp_path, cycle=9, keep=2)
        assert (tmp_path / "notes.txt").exists()

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert remove_old_state_files(tmp_path / "absent", cycle=1, keep=2) == 0


def _gisdata():
    """Minimal stand-in for GISData: three active rows, one row outside."""
    dtm = np.full(SHAPE, np.nan)
    dtm[0:3, :] = 100.0
    return SimpleNamespace(grids={"dtm": dtm, "flow_acc": dtm.copy()})


class TestStateMask:
    def test_mask_covers_the_model_domain(self):
        mask = build_state_mask(_gisdata())
        assert mask.shape == SHAPE
        assert mask[0:3, :].all()
        assert not mask[3, :].any()

    def test_mask_is_the_union_of_the_two_domains(self):
        """A cell finite in either grid must be stored, or its value would be lost."""
        gisdata = _gisdata()
        gisdata.grids["flow_acc"] = np.full(SHAPE, np.nan)
        gisdata.grids["flow_acc"][3, 0] = 5.0

        mask = build_state_mask(gisdata)
        assert mask[0, 0]  # inside the DTM domain only
        assert mask[3, 0]  # inside the flow-accumulation domain only
        assert not mask[3, 1]  # outside both

    def test_missing_flow_acc_falls_back_to_the_dtm(self):
        gisdata = _gisdata()
        del gisdata.grids["flow_acc"]

        mask = build_state_mask(gisdata)
        assert mask[0:3, :].all()
        assert not mask[3, :].any()
