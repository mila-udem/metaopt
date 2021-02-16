# -*- coding: utf-8 -*-
"""
=================
Python API basics
=================

This short tutorial will show you the basics to use Oríon in python. We will optimize a simple
1-d ``rosenbrock`` function with random search and TPE and visualize the regret curve to compare
the algorithms.

We first import the only function needed, :func:`build experiment <orion.client.build_experiment>`.
"""
from orion.client import build_experiment

#%%
# We configure the database with PickledDB so that the results are saved locally on disk. This
# enables resuming the experiment and running parallel workers.
#
# .. TODO Replace with just nothing when pickleddb becomes the default.
#

storage = {
    "type": "legacy",
    "database": {
        "type": "pickleddb",
        "host": "./db.pkl",
    },
}

#%%
# We define the search space for the optimization. Here, the optimization algorithm may explore
# real values for ``x`` between 0 and 30 only. See documentation of :ref:`search-space` for more
# information.

space = {"x": "uniform(0, 30)"}

#%%
# We then build the experiment with the name ``random-rosenbrock``. The name is by Oríon as
# an `id` for the experiment. Each experiment must have a unique name.

experiment = build_experiment(
    "random-rosenbrock",
    space=space,
    storage=storage,
)

#%%
# For this example we use a 1-d rosenbrock function. We must return a list of results,
# for Oríon. Results must have the format
# ``{name: <str>: type: <'objective', 'constraint' or 'gradient'>, value=<float>}`` otherwise
# a ``ValueError`` will be raised. At least one of the results must have the type ``objective``,
# the metric that is minimized by the algorithm.


def rosenbrock(x, noise=None):
    """Evaluate partial information of a quadratic."""
    y = x - 34.56789
    z = 4 * y ** 2 + 23.4

    return [{"name": "objective", "type": "objective", "value": z}]


#%%
# We then pass the function ``rosenbrock`` to
# :meth:`workon() <orion.client.experiment.ExperimentClient.workon>`. This method
# will iteratively try new sets of hyperparameters suggested by the optimization algorithm
# until it reaches 20 trials.

experiment.workon(rosenbrock, max_trials=20)

#%%
# Now let's plot the regret curve to see how well went the optimization.

experiment.plot.regret().show()

#%%
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/random-rosenbrock_regret.html
#
# We have here on the x-axis the trials order by suggestion time, that is the order they
# were tried inside the ``workon`` loop. The y-axis is the objective of the trials, the lower the
# better. The red curve represents the regret, the best achieved objective at time `x`, and the
# blue dots are the objectives of the different trials.
#
# Fortunately, we can see looking at the regret curve that the best results converge close to the
# minimum. That means the optimization was at least reasonably good. On the other-hand, looking at
# the trials we see that the algorithm made suggestions far from optimal from the beginning
# to the end of the optimization. That is to be expected, by default Oríon uses random search. All
# trials were randomly sampled. Let's try next a different algorithm to observe a proper convergence
# behavior.
#
# Let's use a
# `Tree Parzen Estimator
# <https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html>`_
# that can easily find the optimal solution. We specify the algorithm configuration t
# :func:`build experiment <orion.client.build_experiment>`

experiment = build_experiment(
    "tpe-rosenbrock",
    space=space,
    algorithms={"tpe": {"n_initial_points": 5}},
    storage=storage,
)

#%%
# We then again run the optimization for 20 trials and plot the regret.

experiment.workon(rosenbrock, max_trials=20)

experiment.plot.regret().show()

# sphinx_gallery_thumbnail_path = '_static/python.png'

#%%
# .. This file is produced by docs/scripts/build_database_and_plots.py
#
# .. raw:: html
#     :file: ../_static/tpe-rosenbrock_regret.html
#
# We can see the stark difference with the results of the random search. After the first 5 random
# points (see tpe's configuration above), the TPE already identified the optimal region for `x`
# and explored this subspace.
#
# Further readings
# ----------------
#
# User guides:
#
# - :ref:`Search space <search-space>`
# - :ref:`Algorithms <Setup Algorithms>`
# - :ref:`Visualizations <visualizations>`
#
# API:
#
# - :func:`orion.client.build_experiment`
# - :func:`orion.client.get_experiment`
# - :class:`orion.client.experiment.ExperimentClient`
