#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.random`."""
import numpy
import pytest

from orion.algo.random import Random
from orion.algo.space import Integer, Real, Space
from orion.testing.algo import BaseAlgoTests


class TestRandomSearch(BaseAlgoTests):
    algo_name = "random"
    config = {"seed": 123456}

    def test_suggest(self, mocker, num, attr):
        """Verify that suggest returns correct number of points."""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest()
        assert len(points) == self.max_trials
        points = algo.suggest()
        assert len(points) == 1


TestRandomSearch.set_phases([("random", 0, "space.sample")])
