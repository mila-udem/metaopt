# -*- coding: utf-8 -*-
"""Generic tests for Space"""
from collections import OrderedDict

from orion.algo.space import Categorical, Dimension, Fidelity, Integer, Real, Space


def dim1():
    """Create an example of `orion.algo.space.Dimension`."""
    dim = Real("yolo0", "norm", 0.9, shape=(3, 2))
    return dim


def dim2():
    """Create a second example of `orion.algo.space.Dimension`."""
    probs = (0.1, 0.2, 0.3, 0.4)
    categories = ("asdfa", "2", "3", "4")
    categories = OrderedDict(zip(categories, probs))
    dim2 = Categorical("yolo2", categories)
    return dim2


def dim3():
    """Create an example of integer `orion.algo.space.Dimension`."""
    return Integer("yolo3", "uniform", 3, 7)


def dim4():
    """Create an example of Fidelity `orion.algo.space.Dimension`."""
    return Fidelity("yolo4", 1, 10)


def build_space(dims=None):
    """Create an example `orion.algo.space.Space`."""
    if dims is None:
        dims = [dim1(), dim2(), dim3(), dim4()]
    space = Space()
    for dim in dims:
        space.register(dim)
    return space
