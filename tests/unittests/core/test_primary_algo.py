#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.base.primary_algo`."""

import pytest

from orion.algo.base import PrimaryAlgo


@pytest.fixture()
def palgo(dumbalgo, space, fixed_suggestion):
    """Set up a PrimaryAlgo with dumb configuration."""
    algo_config = {'DumbAlgo': {
        'value': fixed_suggestion,
        }}
    palgo = PrimaryAlgo(space, algo_config)

    return palgo


class TestPrimaryAlgoWraps(object):
    """Test if PrimaryAlgo is actually wrapping the configured algorithm.

    Does not test for transformations.
    """

    def test_init_and_configuration(self, dumbalgo, palgo, fixed_suggestion):
        """Check if initialization works."""
        assert isinstance(palgo.instance, dumbalgo)
        assert palgo.configuration == {
            'dumbalgo': {
                'seed': None,
                'value': fixed_suggestion,
                'scoring': 0,
                'judgement': None,
                'suspend': False,
                'done': False,
                }
            }

    def test_space_can_only_retrieved(self, palgo, space):
        """Set space is forbidden, getter works as supposed."""
        assert palgo.space == space
        with pytest.raises(AttributeError):
            palgo.space = 5

    def test_suggest(self, palgo, fixed_suggestion):
        """Suggest wraps suggested."""
        assert palgo.suggest() == [fixed_suggestion]
        assert palgo.suggest(4) == [fixed_suggestion] * 4
        palgo.instance.possible_values = [(5,)]
        with pytest.raises(AssertionError):
            palgo.suggest()

    def test_observe(self, palgo, fixed_suggestion):
        """Observe wraps observations."""
        palgo.observe([fixed_suggestion], [5])
        assert palgo.instance._points == [fixed_suggestion]
        assert palgo.instance._results == [5]
        with pytest.raises(AssertionError):
            palgo.observe([fixed_suggestion], [5, 8])
        with pytest.raises(AssertionError):
            palgo.observe([(5,)], [5])

    def test_isdone(self, palgo):
        """Wrap isdone."""
        palgo.instance.done = 10
        assert palgo.is_done == 10
        assert palgo.instance._times_called_is_done == 1

    def test_shouldsuspend(self, palgo):
        """Wrap should_suspend."""
        palgo.instance.suspend = 55
        assert palgo.should_suspend == 55
        assert palgo.instance._times_called_suspend == 1

    def test_score(self, palgo, fixed_suggestion):
        """Wrap score."""
        palgo.instance.scoring = 60
        assert palgo.score(fixed_suggestion) == 60
        assert palgo.instance._score_point == fixed_suggestion
        with pytest.raises(AssertionError):
            palgo.score((5,))

    def test_judge(self, palgo, fixed_suggestion):
        """Wrap judge."""
        palgo.instance.judgement = 'naedw'
        assert palgo.judge(fixed_suggestion, 8) == 'naedw'
        assert palgo.instance._judge_point == fixed_suggestion
        assert palgo.instance._measurements == 8
        with pytest.raises(AssertionError):
            palgo.judge((5,), 8)


class TestPrimaryAlgoTransforms(object):
    """Check if PrimaryAlgo appropriately transforms spaces and samples."""

    pass
