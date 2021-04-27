#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Task for Branin Function
=========================
"""
import copy
import math

import numpy

from orion.benchmark.task.base import BaseTask


class Branin(BaseTask):
    """`Branin function <http://infinity77.net/global_optimization/test_functions_nd_B.html#go_benchmark.Branin01>`_
    as benchmark task
    """

    def __init__(self, max_trials=20):
        super(Branin, self).__init__(max_trials=max_trials)

    def call(self, x):
        """Evaluate a 2-D branin function."""
        x = copy.deepcopy(x)
        x[0] = (x[0] * 15) - 5
        x[1] = x[1] * 15

        a = 1
        b = 5.1 / (4 * numpy.square(math.pi))
        c = 5 / math.pi
        r = 6
        t = 1 / (8 * math.pi)
        s = 10
        y = (
            a * numpy.square(x[1] - b * numpy.square(x[0]) + c * x[0] - r)
            + s * (1 - t) * numpy.cos(x[0])
            + s
        )

        return [dict(name="branin", type="objective", value=y)]

    def get_search_space(self):
        """Return the search space for the task objective function"""
        rspace = {"x": "uniform(0, 1, shape=2, precision=10)"}

        return rspace
