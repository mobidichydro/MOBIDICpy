"""Tests for parameter mapping (dot-notation path traversal)."""

import pytest

from mobidic.calibration.parameter_mapping import (
    resolve_dot_path,
    set_dot_path,
    read_model_input_csv,
)


class TestResolveDotPath:
    def test_resolve_dict(self):
        d = {"a": {"b": {"c": 42}}}
        assert resolve_dot_path(d, "a.b.c") == 42

    def test_resolve_single_level(self):
        d = {"x": 10}
        assert resolve_dot_path(d, "x") == 10

    def test_resolve_missing_key(self):
        d = {"a": {"b": 1}}
        with pytest.raises(KeyError, match="missing"):
            resolve_dot_path(d, "a.missing")

    def test_resolve_pydantic_model(self):
        from pydantic import BaseModel

        class Inner(BaseModel):
            value: float = 3.14

        class Outer(BaseModel):
            inner: Inner = Inner()

        obj = Outer()
        assert resolve_dot_path(obj, "inner.value") == 3.14

    def test_resolve_nested_dict_deep(self):
        d = {"l1": {"l2": {"l3": {"l4": "deep"}}}}
        assert resolve_dot_path(d, "l1.l2.l3.l4") == "deep"


class TestSetDotPath:
    def test_set_existing_key(self):
        d = {"a": {"b": {"c": 1}}}
        set_dot_path(d, "a.b.c", 99)
        assert d["a"]["b"]["c"] == 99

    def test_set_single_level(self):
        d = {"x": 0}
        set_dot_path(d, "x", 42)
        assert d["x"] == 42

    def test_set_missing_intermediate_key(self):
        d = {"a": {"b": 1}}
        with pytest.raises(KeyError, match="missing"):
            set_dot_path(d, "a.missing.c", 99)

    def test_set_new_leaf_key(self):
        d = {"a": {"b": {}}}
        set_dot_path(d, "a.b.new_key", "hello")
        assert d["a"]["b"]["new_key"] == "hello"


class TestReadModelInputCsv:
    def test_read_valid_csv(self, tmp_path):
        csv_path = tmp_path / "model_input.csv"
        csv_path.write_text(
            "parameter_key,value\nparameters.multipliers.ks_factor,0.5\nparameters.soil.gamma,1.0e-04\n"
        )
        params = read_model_input_csv(csv_path)
        assert params["parameters.multipliers.ks_factor"] == 0.5
        assert params["parameters.soil.gamma"] == pytest.approx(1.0e-4)

    def test_read_empty_csv(self, tmp_path):
        csv_path = tmp_path / "model_input.csv"
        csv_path.write_text("parameter_key,value\n")
        params = read_model_input_csv(csv_path)
        assert params == {}

    def test_read_csv_with_spaces(self, tmp_path):
        csv_path = tmp_path / "model_input.csv"
        csv_path.write_text("parameter_key, value\n parameters.routing.wcel , 3.5 \n")
        params = read_model_input_csv(csv_path)
        assert params["parameters.routing.wcel"] == 3.5
