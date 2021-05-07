# -*- coding: utf-8 -*-
"""
Grid Search
===========
"""
import itertools
import logging

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Categorical, Fidelity, Integer, Real

log = logging.getLogger(__name__)


def grid(dim, num):
    """Build a one-dim grid of num points"""

    if dim.type == "categorical":
        return categorical_grid(dim, num)
    elif dim.type == "integer":
        return discrete_grid(dim, num)
    elif dim.type == "real":
        return real_grid(dim, num)
    elif dim.type == "fidelity":
        return fidelity_grid(dim, num)
    else:
        raise TypeError(
            "Grid Search only supports `real`, `integer`, `categorical` and `fidelity`: "
            f"`{dim.type}`\n"
            "For more information on dimension types, see "
            "https://orion.readthedocs.io/en/stable/user/searchspace.html"
        )


def fidelity_grid(dim, num):
    """Build fidelity grid, that is, only top value"""
    return [dim.interval()[1]]


def categorical_grid(dim, num):
    """Build categorical grid, that is, all categories"""
    categories = dim.interval()
    if len(categories) != num:
        log.warning(
            f"Categorical dimension {dim.name} does not have {num} choices: {categories}. "
            "Will use {len(categories)} choices instead."
        )
    return categories


def discrete_grid(dim, num):
    """Build discretized real grid"""
    grid = real_grid(dim, num)

    _, b = dim.interval()

    discrete_grid = [int(numpy.round(grid[0]))]
    for v in grid[1:]:
        int_v = int(numpy.round(v))
        if int_v <= discrete_grid[-1]:
            int_v = discrete_grid[-1] + 1

        if int_v > b:
            log.warning(
                f"Cannot list {num} discrete values for {dim.name}. "
                "Will use {len(discrete_grid)} points instead."
            )
            break

        discrete_grid.append(int_v)

    return discrete_grid


def real_grid(dim, num):
    """Build real grid"""
    if dim.prior_name.endswith("reciprocal"):
        a, b = dim.interval()
        return list(_log_grid(a, b, num))
    elif dim.prior_name.endswith("uniform"):
        a, b = dim.interval()
        return list(_lin_grid(a, b, num))
    else:
        raise TypeError(
            "Grid Search only supports `loguniform`, `uniform` and `choices`: "
            "`{}`".format(dim.prior_name)
        )


def _log_grid(a, b, num):
    return numpy.exp(_lin_grid(numpy.log(a), numpy.log(b), num))


def _lin_grid(a, b, num):
    return numpy.linspace(a, b, num=num)


class GridSearch(BaseAlgorithm):
    """Grid Search algorithm

    Parameters
    ----------
    n_values: int or dict
        Number of points for each dimensions, or dictionary specifying number of points for each
        dimension independently (name, n_values). For categorical dimensions, n_values will not be
        used, and all categories will be used to build the grid.
    """

    requires_type = None
    requires_dist = None
    requires_shape = "flattened"

    def __init__(self, space, n_values=100, seed=None):
        super(GridSearch, self).__init__(space, n_values=n_values, seed=seed)
        self.n = 0
        self.grid = None

    def _initialize(self):
        """Initialize the grid once the space is transformed"""
        n_values = self.n_values
        if not isinstance(n_values, dict):
            n_values = {name: self.n_values for name in self.space.keys()}

        self.grid = self.build_grid(
            self.space, n_values, getattr(self, "max_trials", 10000)
        )

    @staticmethod
    def build_grid(space, n_values, max_trials=10000):
        """Build a grid of trials

        Parameters
        ----------
        n_values: int or dict
            Number of points for each dimensions, or dictionary specifying number of points for each
            dimension independently (name, n_values). For categorical dimensions, n_values will not be
            used, and all categories will be used to build the grid.
        max_trials: int
            Maximum number of trials for the grid. If n_values lead to more trials than max_trials,
            the n_values will be adjusted down. Will raise ValueError if it is impossible to build
            a grid smaller than max_trials (for instance if choices are too large).

        """
        adjust = 0
        n_trials = float("inf")

        while n_trials > max_trials:
            coordinates = []
            capped_values = []
            for name, dim in space.items():
                capped_value = max(n_values[name] - adjust, 1)
                capped_values.append(capped_value)
                coordinates.append(list(grid(dim, capped_value)))

            if all(value <= 1 for value in capped_values):
                raise ValueError(
                    f"Cannot build a grid smaller than {max_trials}. "
                    "Try reducing the number of choices in categorical dimensions."
                )

            n_trials = numpy.prod([len(dim_values) for dim_values in coordinates])
            # TODO: Use binary search instead of incrementing by one.
            adjust += 1

        if adjust > 1:
            log.warning(
                f"`n_values` reduced by {adjust-1} to limit number of trials below {max_trials}."
            )

        return list(itertools.product(*coordinates))

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super(GridSearch, self).state_dict
        state_dict["grid"] = self.grid
        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        Parameters
        ----------
        state_dict: dict
            Dictionary representing state of an algorithm
        """
        super(GridSearch, self).set_state(state_dict)
        self.grid = state_dict["grid"]

    def suggest(self, num):
        """Return the entire grid of suggestions

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        """
        if self.grid is None:
            self._initialize()
        i = 0
        points = []
        while len(points) < num and i < len(self.grid):
            point = self.grid[i]
            if not self.has_suggested(point):
                self.register(point)
                points.append(point)
            i += 1

        return points

    @property
    def is_done(self):
        """Return True when all grid has been covered."""
        return (
            super(GridSearch, self).is_done
            or self.grid is not None
            and self.n_suggested >= len(self.grid)
        )

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        # NOTE: Override parent method to ignore `seed`
        return {self.__class__.__name__.lower(): {"n_values": self.n_values}}
