#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.strategies`."""
import pytest

from orion.core.worker.strategy import (MaxParallelStrategy,
                                        MeanParallelStrategy,
                                        NoParallelStrategy,
                                        Strategy)
from orion.core.worker.trial import Trial


@pytest.fixture
def observations():
    """10 objective observations"""
    points = [i for i in range(10)]
    results = [{'objective': points[i]} for i in range(10)]

    return points, results


@pytest.fixture
def incomplete_trial():
    """Return a single trial without results"""
    return Trial(params=[{'name': 'a', 'type': 'integer', 'value': 6}])


class TestStrategyFactory:
    """Test creating a parallel strategy with the Strategy class"""

    def test_create_noparallel(self):
        """Test creating a NoParallelStrategy class"""
        strategy = Strategy({'NoParallelStrategy': {}}).instance
        assert isinstance(strategy, NoParallelStrategy)

    def test_create_meanparallel(self):
        """Test creating a MeanParallelStrategy class"""
        strategy = Strategy({'MeanParallelStrategy': {}}).instance
        assert isinstance(strategy, MeanParallelStrategy)


class TestParallelStrategies:
    """Test the different parallel strategy methods"""

    def test_max_parallel_strategy(self, observations, incomplete_trial):
        """Test that MaxParallelStrategy lies using the max"""
        points, results = observations

        strategy = MaxParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)

        max_value = max(result['objective'] for result in results)
        assert lying_result.value == max_value

    def test_mean_parallel_strategy(self, observations, incomplete_trial):
        """Test that MeanParallelStrategy lies using the mean"""
        points, results = observations

        strategy = MeanParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)

        mean_value = (sum(result['objective'] for result in results) /
                      float(len(results)))
        assert lying_result.value == mean_value

    def test_no_parallel_strategy(self, observations, incomplete_trial):
        """Test that NoParallelStrategy lies outputs None"""
        points, results = observations

        strategy = NoParallelStrategy()
        strategy.observe(points, results)
        lying_result = strategy.lie(incomplete_trial)
        assert lying_result is None
