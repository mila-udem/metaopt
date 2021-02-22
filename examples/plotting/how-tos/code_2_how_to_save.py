"""
===========
How to save
===========


Image
-----

Plots can be saved to multiple image formats using ``plotly``'s 
`fig.write_image() <https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_image.html>`_.

"""

from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
fig = experiment.plot.regret()
fig.write_image("regret.png")

#%%
# See `plotly`'s `short tutorial <https://plotly.com/python/static-image-export/>`_
# for more examples.
#
# HTML
# ----
#
# Using ``write_image`` is convenient for static images that can be included in PDF articles
# but it looses all advantages of plotly's interactive plots.
# You can save an HTML version instead using
# `fig.write_html() <https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html>`_
# if you wish to keep an interactive version of the plot.

fig.write_html("regret.html")

#%%
# JSON
# ----
#
# Plots can also be saved to json using
# `fig.to_json() <https://plotly.github.io/plotly.py-docs/generated/plotly.io.to_json.html>`_.
# This is the format used by the :ref:`web-api` for instance.

fig.to_json("regret.json")

# sphinx_gallery_thumbnail_path = '_static/save.png'
