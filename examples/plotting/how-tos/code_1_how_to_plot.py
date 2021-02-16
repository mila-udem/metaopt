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
storage = dict(type="legacy", database=dict(type="pickleddb", host="../../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
experiment.plot.regret()

#%%
# Or with plotting module

import orion.plotting.base as plot

plot.regret(experiment)

# sphinx_gallery_thumbnail_path = '_static/paint.png'
