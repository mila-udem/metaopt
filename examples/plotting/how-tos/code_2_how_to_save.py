"""
===========
How to save
===========

fig.write_image

https://plotly.com/python/static-image-export/

jup notebook?

fig.write_html

https://plotly.com/python/interactive-html-export/

fig.to_json (correct?)

chart studio
"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../database.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("lateral-view-multitask3", storage=storage)
experiment.plot.regret()

#%%
# Or with plotting module

import orion.plotting.base as plot

plot.regret(experiment)

# sphinx_gallery_thumbnail_path = '_static/save.png'
