# -*- coding: utf-8 -*-
"""
Different mutate functions: large-scale evolution of image classifiers
======================================================================

Implement evolution to exploit configurations with fixed resource efficiently

"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def default_mutate(search_space, rng, old_value, **kwargs):
    """Get a default mutate function"""
    multiply_factor = kwargs.pop("multiply_factor", 3.0)
    add_factor = kwargs.pop("add_factor", 1)
    volatility = kwargs.pop("volatility", 0.001)
    if search_space.type == "real":
        lower_bound, upper_bound = search_space.interval()
        factors = (
            1.0 / multiply_factor
            + (multiply_factor - 1.0 / multiply_factor) * rng.random()
        )
        if lower_bound <= old_value * factors <= upper_bound:
            new_value = old_value * factors
        elif lower_bound > old_value * factors:
            new_value = lower_bound + volatility * rng.random()
        else:
            new_value = upper_bound - volatility * rng.random()
    elif search_space.type == "integer":
        print(search_space)
        lower_bound, upper_bound = search_space.interval()
        factors = int(add_factor * (2 * rng.randint(2) - 1))
        if lower_bound <= old_value + factors <= upper_bound:
            new_value = int(old_value) + factors
        elif lower_bound > old_value + factors:
            new_value = int(lower_bound)
        else:
            new_value = int(upper_bound)
    elif search_space.type == "categorical":
        # TODO: This ignores the probabilities passed to search space.
        #       The mutation function should work directly at the search space level
        #       instead of separately on each dimensions. This would make it possible
        #       to sample properly the categorical dimensions.
        new_value = rng.choice(search_space.interval())
    else:
        print(search_space.type)
        new_value = old_value
    return new_value
