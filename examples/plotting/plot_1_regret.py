"""
=============
Regret curves
=============

.. hint:: 

   Conveys how quickly the algorithm found the best hyperparameters.
    
The regret is the difference between the achieved objective and the optimal objective.
Plotted as a time serie, it shows how fast an optimization algorithm approaches the best
objective. An equivalent way to visualize it is using the cumulative minimums instead of the
differences. Oríon plots the cumulative minimums so that the objective can easily be read
from the y-axis.

.. autofunction:: orion.plotting.base.regret
    :noindex:

The regret plot can be executed directly from the ``experiment`` with ``plot.regret()`` as
shown in the example below.

"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("random-rosenbrock", storage=storage)
fig = experiment.plot.regret()
fig

#%%
# The objective of the trials is overlayed as a scatter plot under the regret curve.
# Thanks to this we can see whether the algorithm focused its optimization close to the
# optimum (if all points are close to the regret curve near the end) or if it explored far
# from it (if many points are far from the regret curve near the end). We can see in this example
# with random search that the algorithm unsurpringly randomly explored the space.
# If we plot the results from the algorithm TPE applied on the same task, we will see a very
# different pattern
# (see how the results were generated in tutorial :ref:`sphx_glr_auto_tutorials_code_1_python_api.py`).

experiment = get_experiment("tpe-rosenbrock", storage=storage)
fig = experiment.plot.regret()
fig

#%%
# You can hover your mouse over the trials to see the configuration of the corresponding trials.
# The configuration of the trials is following this format:
#
# ::
#
#    ID: <trial id>
#    value: <objective>
#    time: <time x-axis is ordered by>
#    parameters
#      <name>: <value>

#%%
# For now, the only option to customize the regret plot is ``order_by``, the sorting order
# of the trials on the x-axis. Contributions are more than welcome to increase the customizability
# of the plot!
# By default the sorting order is ``suggested``, the order the trials were suggested by the
# optimization algorithm. Other options are ``reserved``, the time the trials started being executed
# and ``completed``, the time the trials were completed.

fig = experiment.plot.regret(order_by="completed")
fig

#%%
# In this example, the order by ``suggested`` or by ``completed`` is the same,
# but parallelized experiments can lead to different order of completed trials.

# TODO: Add documentation for `regrets()`

#%%
# Finally we save the image to serve as a thumbnail for this example. See
# the guide
# :ref:`How to save <sphx_glr_auto_examples_how-tos_code_2_how_to_save.py>`
# for more information on image saving.

fig.write_image("../../docs/src/_static/regret_thumbnail.png")

# sphinx_gallery_thumbnail_path = '_static/regret_thumbnail.png'
