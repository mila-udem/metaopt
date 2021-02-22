"""
==============================
How to share with chart studio
==============================

This tutorial explains how to share plots using chart studio. This 
method that was used to share plots on the WordPress blog-post:
`Improved Deep Learning Workflows Through Hyperparameter Optimization with Oríon
<https://mila.quebec/en/article/improved-deep-learning-workflows-through-hyperparameter-optimization-with-orion/>`_.

Uploading to Chart Studio
-------------------------

If you do not already have a Chart Studio, you can create a 
`free one <https://plotly.com/api_signup>`_. 
With your account set up, you can get your `API key <https://plotly.com/settings/api>`_
set Chart Studio's credentials. 

"""
import chart_studio

username = "<your username>"
api_key = "<your api_key>"

chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

#%%
# This will create a credential file locally that will be used in next step.
# Next you can create a ``figure`` and save it to Chart Studio.

# # Push your visualiztion to your account using the following lines of code:
from orion.client import get_experiment

# Specify the database where the experiments are stored. We use a local PickleDB here.
storage = dict(type="legacy", database=dict(type="pickleddb", host="../../db.pkl"))

fig = experiment.plot.regret()


import chart_studio.plotly as py

py.plot(fig, filename="regret")

# sphinx_gallery_thumbnail_path = '_static/share.png'

#%%
# The plot will be saved under the name ``regret`` in your
# `Chart Studio profile <https://chart-studio.plotly.com/organize/home>`_.
# Here is one example that was used in the blog-post mentioned earlier:
# `https://chart-studio.plotly.com/~xavier.bouthillier/1 <https://chart-studio.plotly.com/~xavier.bouthillier/1>`_.

#%%
# Sharing on WordPress
# --------------------
#
# Saving HTML version of plots
# does not work well with WordPress because it includes JavaScript.
# Using Chart Studio makes it possible to embed extarnal URL in WordPress posts.
#
# With Chart Studio,
# you can get an sharing URL for an embedded plot. In the Viewer page of Chart Studio,
# click on `export` and then `Embed URL`. This should bring you to a page with an URL
# formatted as ``https://chart-studio.plotly.com/~<your username>/<plot id>.embed``.
#
# With the `WordPress plugin iframe <https://fr.wordpress.org/plugins/iframe/>`_, you
# can embed your plot in a blog post. For example, the first plot in
# our blog-post is embedded with the following snippet.
#
# ::
#
#     [iframe src="//plotly.com/~xavier.bouthillier/1.embed?link=false&amp;autosize=true&amp;height=350"]
#
# Arguments to customized the embedded are documented
# `here
# <https://plotly.com/chart-studio-help/embed-graphs-in-websites/#step-8-customize-the-iframe>`_.
# See `starting guide <https://plotly.com/python/getting-started-with-chart-studio/>`_
# for more information on Chart Studio.
