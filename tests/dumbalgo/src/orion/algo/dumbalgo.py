# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.dumbalgo` -- This is a dumb algo for test purposes
====================================================================

.. module:: dumbalgo
   :platform: Unix
   :synopsis: Test algo

"""
from orion.algo.base import BaseAlgorithm


class DumbAlgo(BaseAlgorithm):
    """Stab class for `BaseAlgorithm`."""

    def __init__(self, space, value=5,
                 scoring=0, judgement=None,
                 suspend=False, done=False, seed=None):
        """Configure returns, allow for variable variables."""
        self._times_called_suspend = 0
        self._times_called_is_done = 0
        self._num = 0
        self._index = 0
        self._points = []
        self._results = []
        self._score_point = None
        self._judge_point = None
        self._measurements = None
        self.possible_values = [value]
        super(DumbAlgo, self).__init__(space, value=value,
                                       scoring=scoring, judgement=judgement,
                                       suspend=suspend,
                                       done=done,
                                       seed=seed)

    def seed(self, seed):
        """Set the index to seed.

        Setting the seed as an index so that unit-tests can force the algorithm to suggest the same
        values as if seeded.
        """
        self._index = seed if seed is not None else 0

    def suggest(self, num=1):
        """Suggest based on `value`."""
        self._num += num

        rval = []
        while len(rval) < num:
            value = self.possible_values[min(self._index, len(self.possible_values) - 1)]
            self._index += 1
            rval.append(value)

        self._suggested = rval

        return rval

    def observe(self, points, results):
        """Log inputs."""
        self._points += points
        self._results += results

    def score(self, point):
        """Log and return stab."""
        self._score_point = point
        return self.scoring

    def judge(self, point, measurements):
        """Log and return stab."""
        self._judge_point = point
        self._measurements = measurements
        return self.judgement

    @property
    def should_suspend(self):
        """Cound how many times it has been called and return `suspend`."""
        self._times_called_suspend += 1
        return self.suspend

    @property
    def is_done(self):
        """Cound how many times it has been called and return `done`."""
        self._times_called_is_done += 1
        return self.done
