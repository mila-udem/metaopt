"""
=============
Regret curves
=============

Regret curve documentation...
"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="./database.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("lateral-view-multitask3", storage=storage)
fig = experiment.plot.regret()
fig.write_image("../../docs/src/_static/regret_thumbnail.png")
fig

#%%
# In addition, we can test the same thing again :D.

fig = experiment.plot.regret()
fig

#%%
# and again :D.

fig = experiment.plot.regret()
fig

# sphinx_gallery_thumbnail_path = '_static/regret_thumbnail.png'
