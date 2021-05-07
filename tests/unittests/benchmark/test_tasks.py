#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.benchmark.task`."""

from orion.benchmark.task import Branin, CarromTable, EggHolder, RosenBrock


class TestBranin:
    """Test benchmark task branin"""

    def test_creation(self):
        """Test creation"""
        branin = Branin(2)
        assert branin.max_trials == 2
        assert branin.configuration == {"Branin": {"max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        branin = Branin(2)

        assert callable(branin)

        objectives = branin([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        branin = Branin(2)

        assert branin.get_search_space() == {
            "x": "uniform(0, 1, shape=2, precision=10)"
        }


class TestCarromTable:
    """Test benchmark task CarromTable"""

    def test_creation(self):
        """Test creation"""
        branin = CarromTable(2)
        assert branin.max_trials == 2
        assert branin.configuration == {"CarromTable": {"max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        carrom = CarromTable(2)

        assert callable(carrom)

        objectives = carrom([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        carrom = CarromTable(2)

        assert carrom.get_search_space() == {"x": "uniform(-10, 10, shape=2)"}


class TestEggHolder:
    """Test benchmark task EggHolder"""

    def test_creation(self):
        """Test creation"""
        branin = EggHolder(max_trials=2, dim=3)
        assert branin.max_trials == 2
        assert branin.configuration == {"EggHolder": {"dim": 3, "max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        egg = EggHolder(2)

        assert callable(egg)

        objectives = egg([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        egg = EggHolder(2)

        assert egg.get_search_space() == {"x": "uniform(-512, 512, shape=2)"}


class TestRosenBrock:
    """Test benchmark task RosenBrock"""

    def test_creation(self):
        """Test creation"""
        branin = RosenBrock(max_trials=2, dim=3)
        assert branin.max_trials == 2
        assert branin.configuration == {"RosenBrock": {"dim": 3, "max_trials": 2}}

    def test_call(self):
        """Test to get task function"""
        rb = RosenBrock(2)

        assert callable(rb)

        objectives = rb([1, 2])
        assert type(objectives[0]) == dict

    def test_search_space(self):
        """Test to get task search space"""
        rb = RosenBrock(2)
        assert rb.get_search_space() == {"x": "uniform(-5, 10, shape=2)"}
