"""
====================
Partial Dependencies
====================

.. hint:: 

   Conveys a broad overview of the search space and what has been explored during the experiment.
   Helps identifying best optimal regions of the space.

The partial dependency computes the average predicted performance
with respect to a set of hyperparameters, marginalizing out the other hyperparameters.

To predict the performance on unseen set of hyperparameters, we train a regression model on
available trial history. We build a grid of `g` points for a hyperparameter of interest, or a 2-D
grid of `g ^ 2` points for a pair of hyperparameters.  We sample a group of `n` set of
hyperparameters from the entire space to marginalize over the other hyperparameters. For each value
of the grid, we compute the prediction
of the regression model on all points of the group, with the hyperparameter of interest set to a
value of the grid. For instance, for a 1-D grid of `g` points and a group of `n` points, we compute
`g * n` predictions.

For a search space of `d` hyperparameters, the partial dependency plot is organized as a matrix of 
`(d, d)` subplots. The subplots on the diagonal show the partial dependency of each hyperparameters
separately, while the subplots below the diagonal show the partial dependency between two
hyperparameters. Let's look at a simple example to make it more concrete.

.. autofunction:: orion.plotting.base.partial_dependencies
    :noindex:

The partial dependencies plot can be executed directly from the ``experiment`` with
``plot.partial_dependencies()`` as shown in the example below.

"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
fig = experiment.plot.partial_dependencies()
fig

#%%
# For the plots on the diagonal, the y-axis is the objective and the x-axis is the value of the
# corresponding hyperparameter. For the contour plots below the diagonal, the y-axis and x-axis are
# the values of the corresponding hyperparameters labelled on the left and at the bottom. The
# objective is represented as a color gradient in the contour plots. The light blue area in
# the plots on the diagonal represents the standard deviation of the predicted objective when
# varying the other hyperparameters over the search space.
# The black dots represents the
# trials in the current history of the experiment. If you hover your cursor over one dot, you will
# see the configuration of the corresponding trial following this format:
#
# ::
#
#    ID: <trial id>
#    value: <objective>
#    time: <completed time>
#    parameters
#      <name>: <value>
#
# Even for a simple 2-d search space, the partial dependency is very useful. We see very cleary in this example
# the optimal regions for both hyperparameters and we can see as well that the optimal region for
# learning rates is larger when the dropout is low, and narrower when dropout approaches 0.5.
#
# .. TODO::
#
#     Make one toy example where two HPs are dependent.
#

#%%
# Options
# -------
#
# Params
# ~~~~~~
#
# The simple example involved only 2 hyperparameters, but typical search spaces can be much larger.
# The partial dependency plot becomes hard to read with more than 3-5 hyperparameters dependency on
# the size of your screen. With a fix width like in this documentation, 5 hyperparameters are
# impossible to read as you can see below.
# (Data coming from tutorial :ref:`sphx_glr_auto_tutorials_code_2_hyperband_checkpoint.py`)

experiment = get_experiment("hyperband-cifar10", storage=storage)
experiment.plot.partial_dependencies()

#%%
# You can select the hyperparameters to show with the argument ``params``.

experiment.plot.partial_dependencies(params=["gamma", "learning_rate"])

#%%
# Grid resolution
# ~~~~~~~~~~~~~~~
# The grid used for the partial dependency can be more or less coarse. Coarser grids
# will be faster to compute.
import time

experiment = get_experiment("2-dim-exp", storage=storage)
start = time.clock()
fig = experiment.plot.partial_dependencies(n_grid_points=5)
print(time.clock() - start, "seconds to compute")
fig

#%%
# With more points the grid is finer.
start = time.clock()
fig = experiment.plot.partial_dependencies(n_grid_points=50)
print(time.clock() - start, "seconds to compute")
fig

#%%
# Dependency approximation
# ~~~~~~~~~~~~~~~~~~~~~~~~
# By default, the hyperparameters are marginalized over 50 points. This may be suitable
# for a small 2-D search space but likely unsufficient for 5 dimensions or more.
# Here is an example with only 5 samples.

start = time.clock()
fig = experiment.plot.partial_dependencies(n_samples=5)
print(time.clock() - start, "seconds to compute")
fig

#%%
# And now with 200 samples.

start = time.clock()
fig = experiment.plot.partial_dependencies(n_samples=200)
print(time.clock() - start, "seconds to compute")
fig

#%%
# Special cases
# -------------
#
# Logarithmic scale
# ~~~~~~~~~~~~~~~~~
#
# Dimensions with a logarithmic prior :ref:`search-space-loguniform` are linearized before being
# passed to the regression model (using log(dim) instead of dim directly). This means the model is
# trained and will be making predictions in the linearized space. The data is presented in
# logarithmic scale to the user, with the axis adjusted to log scale as well.
#
# Fidelity
# ~~~~~~~~
#
# The fidelity is considered as a logarithmic scale as well. See above for more information on how
# it is handled.
#
# Dimension with shape
# ~~~~~~~~~~~~~~~~~~~~
#
# If some dimensions have a :ref:`search-space-shape` larger than 1, they will be flattened so that
# each subdimension can be represented separately in the subplots.

# Load the data for the specified experiment
experiment = get_experiment("2-dim-shape-exp", storage=storage)
experiment.plot.partial_dependencies()

#%%
# In the example above, the dimension ``learning_rate~loguniform(1e-5, 1e-2, shape=3)``
# is flattened and represented with ``learning_rate[i]``. If the shape would be or more dimensions
# (ex: ``(3, 2)``), the indices would be ``learning_rate[i,j]`` with i=0..2 and j=0..1.

#%%
# The flattened hyperparameters can be fully selected with ``params=['<name>']``.
#

experiment.plot.partial_dependencies(params=["/learning_rate"])

#%%
# Or a subset of the flattened hyperparameters can be selected with ``params=['<name>[index]']``.
#

experiment.plot.partial_dependencies(params=["/learning_rate[0]", "/learning_rate[1]"])


#%%
# Categorical dimension
# ~~~~~~~~~~~~~~~~~~~~~
#
# Categorical dimensions are converted into integer values, so that the regression model
# can handle them. The integers are simply indices that are assigned to each category in arbitrary
# order. Here is an example where dimension ``mt-join`` has the prior
# ``choices(['mean', 'max', 'concat'])``.

# Load the data for the specified experiment
experiment = get_experiment("3-dim-cat-shape-exp", storage=storage)
experiment.plot.partial_dependencies(params=["/learning_rate[0]", "/mt-join"])

#%%
# .. Note::
#
#     For now categorical are plotted using a scatter plot, but a heat-map
#     would be more readable. Contributions are more than welcome! :)

#%%
# Finally we save the image to serve as a thumbnail for this example. See
# the guide
# :ref:`How to save <sphx_glr_auto_examples_how-tos_code_2_how_to_save.py>`
# for more information on image saving.

fig.write_image("../../docs/src/_static/par_dep_thumbnail.png")

# sphinx_gallery_thumbnail_path = '_static/par_dep_thumbnail.png'
