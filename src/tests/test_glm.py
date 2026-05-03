import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from datashield_base import ModelsClient


class MockSession:
    def __init__(self):
        self.calls = []

    def aggregate(self, expr: str):
        self.calls.append(expr)
        if expr.startswith("glmDS1("):
            return {
                "server1": [
                    ["meta", 2],
                    ["(Intercept)", "x"],
                    0,
                    [0, 0],
                    0,
                    0,
                    0,
                    "No errors",
                ]
            }
        if expr.startswith("glmDS2("):
            # Use one stable iteration profile that converges after 2 iterations.
            if len([c for c in self.calls if c.startswith("glmDS2(")]) == 1:
                return {
                    "server1": {
                        "info.matrix": [[4.0, 0.0], [0.0, 9.0]],
                        "score.vect": [2.0, -3.0],
                        "dev": 100.0,
                        "Nvalid": 10,
                        "Nmissing": 0,
                        "Ntotal": 10,
                        "numsubs": 10,
                        "family": {"family": "gaussian", "link": "identity"},
                        "disclosure.risk": 0,
                        "errorMessage": "No errors",
                    }
                }
            return {
                "server1": {
                    "info.matrix": [[4.0, 0.0], [0.0, 9.0]],
                    "score.vect": [0.0, 0.0],
                    "dev": 100.0,
                    "Nvalid": 10,
                    "Nmissing": 0,
                    "Ntotal": 10,
                    "numsubs": 10,
                    "family": {"family": "gaussian", "link": "identity"},
                    "disclosure.risk": 0,
                    "errorMessage": "No errors",
                }
            }
        raise AssertionError(f"Unexpected expression: {expr}")


class MockSessionNamedMeta(MockSession):
    def aggregate(self, expr: str):
        self.calls.append(expr)
        if expr.startswith("glmDS1("):
            return {
                "server1": {
                    "value": [
                        {"num.par.glm": 2},
                        ["(Intercept)", "x"],
                        0,
                        [0, 0],
                        0,
                        0,
                        0,
                        "No errors",
                    ]
                }
            }
        return super().aggregate(expr)


def test_get_glm_requires_formula():
    client = ModelsClient(MockSession())
    with pytest.raises(ValueError, match="regression formula"):
        client.get_glm(formula=None, family="gaussian")


def test_get_glm_requires_family():
    client = ModelsClient(MockSession())
    with pytest.raises(ValueError, match="family"):
        client.get_glm(formula="y ~ x", family=None)


def test_get_glm_runs_and_returns_expected_shape():
    session = MockSession()
    session.id = "test-session"
    client = ModelsClient(session)

    result = client.get_glm(formula="y ~ x", family="gaussian", maxit=5, CI=0.95)

    assert result is not None
    assert result["iter"] == 2
    assert result["Nvalid"] == 10
    assert result["Nmissing"] == 0
    assert result["Ntotal"] == 10
    assert result["df"] == 8
    assert result["formula"] == "y ~ x"
    assert isinstance(result["coefficients"], list)
    assert len(result["coefficients"]) == 2
    assert result["coefficients"][0]["name"] == "(Intercept)"

    glm_ds1_calls = [c for c in session.calls if c.startswith("glmDS1(")]
    glm_ds2_calls = [c for c in session.calls if c.startswith("glmDS2(")]
    assert len(glm_ds1_calls) == 1
    assert len(glm_ds2_calls) == 2
    assert glm_ds1_calls[0] == "glmDS1(y ~ x, 'gaussian', NULL, NULL, NULL)"
    assert "glmDS2(y ~ x, 'gaussian'" in glm_ds2_calls[0]


def test_get_glm_parses_named_num_par_glm_shape():
    session = MockSessionNamedMeta()
    session.id = "test-session"
    client = ModelsClient(session)

    result = client.get_glm(formula="y ~ x", family="gaussian", maxit=5, CI=0.95)

    assert result is not None
    assert len(result["coefficients"]) == 2
    assert result["coefficients"][0]["name"] == "(Intercept)"


def test_convert_glm2_legacy_format_matches_new_format_fixture():
    client = ModelsClient(MockSession())

    fixtures_dir = Path(__file__).parent / "data"
    legacy = json.loads((fixtures_dir / "glm2-legacy-out.json").read_text(encoding="utf-8"))
    new = json.loads((fixtures_dir / "glm2-out.json").read_text(encoding="utf-8"))

    converted = client._convert_glm2_legacy_format(legacy["server1"])

    expected = dict(new["server1"])
    expected["errorMessage"] = expected.pop("errorMessage2")

    def assert_nested_close(actual, expected_value):
        if isinstance(expected_value, dict):
            assert isinstance(actual, dict)
            assert set(actual.keys()) == set(expected_value.keys())
            for key in expected_value:
                assert_nested_close(actual[key], expected_value[key])
            return

        if isinstance(expected_value, list):
            assert isinstance(actual, list)
            assert len(actual) == len(expected_value)
            for actual_item, expected_item in zip(actual, expected_value, strict=True):
                assert_nested_close(actual_item, expected_item)
            return

        if isinstance(expected_value, (int, float)):
            assert actual == pytest.approx(expected_value, rel=1e-8, abs=1e-8)
            return

        assert actual == expected_value

    assert_nested_close(converted, expected)
