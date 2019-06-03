#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.algo.random`."""
import numpy
import pytest

from orion.algo.random import Random
from orion.algo.space import Integer, Real, Space


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'norm', 0.9)
    space.register(dim2)

    return space


def test_seeding(space):
    """Verify that seeding makes sampling deterministic"""
    random_search = Random(space)

    assert hasattr(random_search, 'rng')
    random_search.seed_rng(1)
    a = random_search.suggest(1)[0]
    assert not numpy.allclose(a, random_search.suggest(1)[0])

    random_search.seed_rng(1)
    assert numpy.allclose(a, random_search.suggest(1)[0])
