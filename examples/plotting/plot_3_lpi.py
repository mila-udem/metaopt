"""
==========================
Local Parameter Importance
==========================

.. hint:: 

   Conveys a very compact measure of the importance of the different hyperparameters
   to achieve the best objective found so far.

The local parameter importance measures the variance of the results when 
varying one hyperparameter and keeping all other fixes [Biedenkapp2018]_.

.. todo::

    TODO explain in more details.

.. [Biedenkapp2018] Biedenkapp, André, et al.
    "Cave: Configuration assessment, visualization and evaluation."
    International Conference on Learning and Intelligent Optimization.
    Springer, Cham, 2018.

"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
fig = experiment.plot.lpi()
fig

#%%
# In addition, we can test the same thing again :D.

fig = experiment.plot.lpi()
fig

#%%
# and again :D.

fig = experiment.plot.lpi()
fig

#%%
# Finally we save the image to serve as a thumbnail for this example. See
# the guide
# :ref:`How to save <sphx_glr_auto_examples_how-tos_code_2_how_to_save.py>`
# for more information on image saving.

fig.write_image("../../docs/src/_static/lpi_thumbnail.png")

# sphinx_gallery_thumbnail_path = '_static/lpi_thumbnail.png'
