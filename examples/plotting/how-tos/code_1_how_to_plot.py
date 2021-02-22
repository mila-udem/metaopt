"""
============
How to plot.
============

There exist multiple ways to plot using Oríon. We will start with the most convenient one, the
experiment object plot accessor.

``experiment.plot`` accessor
----------------------------

.. note::

    It only supports single experiment plots. It does not support ``regrets`` and ``averages``.

It is possible to render most plots directly with 
the :py:class:`ExperimentClient <orion.client.experiment.ExperimentClient>` object.
The accessor ``ExperimentClient.plot`` can be used to plot the results of the experiment.

"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
fig = experiment.plot.regret()
fig.show()

#%%
# ``plotting`` module
# -------------------
#
# The plotting module can used as well to plot experiments.

import orion.plotting.base as plot

fig = plot.regret(experiment)
fig.show()

#%%
# The advantage of the plotting module is that it can render plots using
# multiple examples as shown in the example below.
#
# .. todo::
#
#     Add example with regrets using multiple experiments from benchmark experiment.
#

fig = plot.regret(experiment)
fig.show()

# sphinx_gallery_thumbnail_path = '_static/paint.png'

#%%
# Command line
# ------------
#
# .. note::
#
#     The plotting command line utility does not support arguments for now. Contributions are
#     welcome! :)
#
# .. note::
#
#     Only supports single experiment plots. It does not support ``regrets`` and ``averages``.
#
# ``orion plot --name <exp> <plot type>``
#
# ::
#
#     orion plot --name 2-dim-exp regret
#
# .. TODO::
#
#     Complete the plot command line
#
# .. TODO::
#
#     Add documentation of the commmand line

#%%
#
# Web API
# -------
#
# .. note::
#
#     Only supports single experiment plots. It does not support ``regrets`` and ``averages``.
#
# The :ref:`web-api` supports queries for single experiment plots. See documentation for
# all queries. The JSON output is generated automatically according to the
# `Plotly.js schema reference <https://plotly.com/python/reference/index/>`_.
