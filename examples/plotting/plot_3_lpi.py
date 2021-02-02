"""
==========================
Local Parameter Importance
==========================

Local Parameter Importance documentation...
"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="./database.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("lateral-view-multitask3", storage=storage)
fig = experiment.plot.lpi()
fig.write_image("../../docs/src/_static/lpi_thumbnail.png")
fig

#%%
# In addition, we can test the same thing again :D.

fig = experiment.plot.lpi()
fig

#%%
# and again :D.

fig = experiment.plot.lpi()
fig

# sphinx_gallery_thumbnail_path = '_static/lpi_thumbnail.png'
