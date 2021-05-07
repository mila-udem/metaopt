#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple example to fill db with python api"""
from orion.client import create_experiment

create_experiment(
    "hunt-python",
    space={"x": "uniform(-50,50)"},
    algorithms={"random": {"seed": 1}},
    max_trials=10,
)

create_experiment(
    "hunt-python-branch-old",
    space={"x": "uniform(-50,50)"},
    algorithms={"random": {"seed": 1}},
    branching={"branch_from": "hunt-python"},
)
