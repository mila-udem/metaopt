"""
============
How to plot.
============

How to plot...

plot.<method>

exp.plot.<method>

REST server

"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
import os

raise RuntimeError(os.getcwd())
storage = dict(type="legacy", database=dict(type="pickleddb", host="../database.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("lateral-view-multitask3", storage=storage)
experiment.plot.regret()

#%%
# Or with plotting module

import orion.plotting.base as plot

plot.regret(experiment)

# sphinx_gallery_thumbnail_path = '_static/paint.png'
