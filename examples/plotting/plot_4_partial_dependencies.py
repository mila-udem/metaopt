"""
====================
Partial Dependencies
====================

.. hint:: 

   Conveys a broad overview of the search space and what has been explored during the experiment.
   Helps identifying best optimal regions of the space.


Partial dependencies documentation...
"""
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../db.pkl"))

# Load the data for the specified experiment
experiment = get_experiment("2-dim-exp", storage=storage)
fig = experiment.plot.partial_dependencies(
    #     params=["/learning_rate[0]", "/learning_rate[1]"]
)
fig

#%%
# In addition, we can test the same thing again :D.

fig = experiment.plot.partial_dependencies()
fig

#%%
# and again :D.

fig = experiment.plot.partial_dependencies()
fig

#%%
# Finally we save the image to serve as a thumbnail for this example. See
# the guide
# :ref:`How to save <sphx_glr_auto_examples_how-tos_code_2_how_to_save.py>`
# for more information on image saving.

fig.write_image("../../docs/src/_static/par_dep_thumbnail.png")

# sphinx_gallery_thumbnail_path = '_static/par_dep_thumbnail.png'
