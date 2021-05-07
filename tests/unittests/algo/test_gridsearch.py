"""Example usage and tests for :mod:`orion.algo.gridsearch`."""
import copy
import logging

import numpy.testing
import pytest

from orion.algo.gridsearch import (
    GridSearch,
    categorical_grid,
    discrete_grid,
    grid,
    real_grid,
)
from orion.algo.space import Categorical, Integer, Real, Space
from orion.testing.algo import BaseAlgoTests, phase


def test_categorical_grid():
    """Test that categorical grid returns all choices"""
    dim = Categorical("yolo", "abcde")
    assert categorical_grid(dim, 5) == dim.categories


def test_too_large_categorical_grid(caplog):
    """Test that large categorical grid returns all choices"""
    dim = Categorical("yolo", "abcde")
    with caplog.at_level(logging.WARNING, logger="orion.algo.gridsearch"):
        assert categorical_grid(dim, 3) == dim.categories
    assert "Categorical dimension yolo does not have 3 choices" in caplog.text


def test_too_small_categorical_grid(caplog):
    """Test that small categorical grid returns all choices"""
    dim = Categorical("yolo", "abcde")
    with caplog.at_level(logging.WARNING, logger="orion.algo.gridsearch"):
        assert categorical_grid(dim, 7) == dim.categories
    assert "Categorical dimension yolo does not have 7 choices" in caplog.text


def test_discrete_grid():
    """Test discrete grid"""
    dim = Integer("yolo", "uniform", -5, 10)
    assert discrete_grid(dim, 6) == [-5, -3, -1, 1, 3, 5]
    assert discrete_grid(dim, 7) == [-5, -3, -2, 0, 2, 3, 5]


def test_too_small_distrete_grid():
    """Test that small discrete grid does not lead to duplicates"""
    dim = Integer("yolo", "uniform", -2, 4)
    assert discrete_grid(dim, 3) == [-2, 0, 2]
    assert discrete_grid(dim, 5) == [-2, -1, 0, 1, 2]
    assert discrete_grid(dim, 50) == [-2, -1, 0, 1, 2]


def test_log_discrete_grid():
    """Test log discrete grid"""
    dim = Integer("yolo", "reciprocal", 1, 1000)
    assert discrete_grid(dim, 4) == [1, 10, 100, 1000]
    assert discrete_grid(dim, 6) == [1, 4, 16, 63, 251, 1000]


def test_unsupported_discrete_prior():
    """Test unsupported discrete prior message"""
    dim = Integer("yolo", "norm", 0, 1)
    with pytest.raises(TypeError) as exc:
        discrete_grid(dim, 6)

    assert exc.match("Grid Search only supports `loguniform`, `uniform`")


def test_real_grid():
    """Test linear real grid"""
    dim = Real("yolo", "uniform", -5, 10)
    assert real_grid(dim, 6) == [-5, -3, -1, 1, 3, 5]
    assert real_grid(dim, 7) == [-5 + 10 / 6.0 * i for i in range(7)]


def test_log_real_grid():
    """Test logarithmic real grid"""
    dim = Real("yolo", "reciprocal", 0.0001, 1)
    assert numpy.allclose(real_grid(dim, 5), [0.0001, 0.001, 0.01, 0.1, 1])
    assert real_grid(dim, 6) == [
        numpy.exp(numpy.log(0.0001) + (numpy.log(1) - numpy.log(0.0001)) / 5 * i)
        for i in range(6)
    ]


def test_unsupported_real_prior():
    """Test unsupported real prior message"""
    dim = Real("yolo", "norm", 0.0001, 1)
    with pytest.raises(TypeError) as exc:
        real_grid(dim, 5)

    assert exc.match("Grid Search only supports `loguniform`, `uniform`")


def test_unsupported_type():
    """Test unsupported real prior message"""

    class NewDimType:
        @property
        def type(self):
            return "newdimtype"

    dim = NewDimType()
    with pytest.raises(TypeError) as exc:
        grid(dim, 5)

    assert exc.match(
        f"Grid Search only supports `real`, `integer`, `categorical` and `fidelity`: `{dim.type}`"
    )


def test_build_grid():
    """Test that grid search builds the proper grid"""
    dim1 = Real("dim1", "uniform", 0, 1)
    dim2 = Integer("dim2", "uniform", 0, 10)
    dim3 = Categorical("dim3", "abcde")
    space = Space()
    space.register(dim1)
    space.register(dim2)
    space.register(dim3)

    grid = GridSearch.build_grid(space, {"dim1": 3, "dim2": 4, "dim3": 1})
    assert len(grid) == 3 * 4 * 5


def test_build_grid_limit_size(caplog):
    """Test that grid search reduces the n_values when grid is too large"""
    dim1 = Real("dim1", "uniform", 0, 1)
    dim2 = Integer("dim2", "uniform", 0, 10)
    dim3 = Categorical("dim3", "abcde")
    space = Space()
    space.register(dim1)
    space.register(dim2)
    space.register(dim3)

    with caplog.at_level(logging.WARNING, logger="orion.algo.gridsearch"):
        grid = GridSearch.build_grid(space, {k: 5 for k in space}, 100)
    assert len(grid) == 4 * 4 * 5

    assert "`n_values` reduced by 1 to limit number of trials below 100" in caplog.text


def test_build_grid_cannot_limit_size(caplog):
    """Test that when choices are too large GridSearch raises ValueError"""
    dim1 = Real("dim1", "uniform", 0, 1)
    dim2 = Integer("dim2", "uniform", 0, 10)
    dim3 = Categorical("dim3", "abcde")
    dim4 = Categorical("dim4", "abcde")
    space = Space()
    space.register(dim1)
    space.register(dim2)
    space.register(dim3)
    space.register(dim4)

    with pytest.raises(ValueError) as exc:
        GridSearch.build_grid(space, {k: 5 for k in space}, 10)

    assert exc.match(
        "Cannot build a grid smaller than 10. "
        "Try reducing the number of choices in categorical dimensions."
    )


class TestGridSearch(BaseAlgoTests):
    algo_name = "gridsearch"
    config = {"n_values": 10}

    @phase
    def test_suggest_lots(self, mocker, num, attr):
        """Test that gridsearch returns the whole grid when requesting more points"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(10000)
        assert len(points) == len(algo.algorithm.grid)

    @pytest.mark.skip(reason="Deterministic algorithm")
    def test_seed_rng(self, mocker, num, attr):
        pass


TestGridSearch.set_phases([("grid", 0, "suggest")])
